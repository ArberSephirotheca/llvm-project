(() => {
  const threadsInput = document.getElementById("threadsInput");
  const seedInput = document.getElementById("seedInput");
  const programSelect = document.getElementById("programSelect");
  const collectiveCfInput = document.getElementById("collectiveCfInput");
  const generateBtn = document.getElementById("generateBtn");
  const runStatus = document.getElementById("runStatus");
  const speedInput = document.getElementById("speedInput");
  const speedLabel = document.getElementById("speedLabel");
  const eventCountEl = document.getElementById("eventCount");
  const eventIndexEl = document.getElementById("eventIndex");
  const laneTable = document.getElementById("laneTable");
  const eventLog = document.getElementById("eventLog");
  const resetBtn = document.getElementById("resetBtn");
  const backBtn = document.getElementById("backBtn");
  const stepBtn = document.getElementById("stepBtn");
  const runBtn = document.getElementById("runBtn");
  const canvas = document.getElementById("timelineCanvas");
  const tooltip = document.getElementById("tooltip");
  const programText = document.getElementById("programText");

  const palette = {
    step: "#4f8bff",
    suspend: "#f97316",
    resume: "#30c9c9",
    return: "#94a3b8",
    control: "#f43f5e",
    unknown: "#9aa4b2",
  };

  let events = [];
  let currentIndex = -1;
  let laneCount = parseInt(threadsInput.value, 10) || 4;
  let speed = parseInt(speedInput.value, 10) || 24;
  let running = false;
  let timer = null;
  let eventPoints = [];

  function fixLegacyLine(line) {
    return line
      .replace(/"active":"mask":"/g, "\"active\":\"")
      .replace(/"expected":"mask":"/g, "\"expected\":\"");
  }

  function extractString(line, key) {
    const match = line.match(new RegExp(`"${key}":"([^"]*)"`));
    return match ? match[1] : "";
  }

  function extractNumber(line, key) {
    const match = line.match(new RegExp(`"${key}":(-?\\d+)`));
    return match ? Number(match[1]) : null;
  }

  function extractBool(line, key) {
    const match = line.match(new RegExp(`"${key}":(true|false)`));
    return match ? match[1] === "true" : null;
  }

  function extractMask(line, key) {
    const legacy = line.match(new RegExp(`"${key}":"mask":"([^"]+)"`));
    if (legacy)
      return legacy[1];
    return extractString(line, key);
  }

  function parseLine(line) {
    const trimmed = line.trim();
    if (!trimmed)
      return null;
    try {
      return JSON.parse(trimmed);
    } catch (err) {
      const fixed = fixLegacyLine(trimmed);
      try {
        return JSON.parse(fixed);
      } catch (err2) {
        const parsed = {
          t: extractNumber(trimmed, "t"),
          event: extractString(trimmed, "event"),
          wave: extractNumber(trimmed, "wave"),
          lane: extractNumber(trimmed, "lane"),
          op: extractString(trimmed, "op"),
          effect: extractString(trimmed, "effect"),
          active: extractMask(trimmed, "active"),
          expected: extractMask(trimmed, "expected"),
          blockSeq: extractNumber(trimmed, "blockSeq"),
          blockKind: extractString(trimmed, "blockKind"),
          blockAddr: extractString(trimmed, "blockAddr"),
          hasValue: extractBool(trimmed, "hasValue"),
        };
        return parsed.event ? parsed : null;
      }
    }
  }

  function normalizeMask(value) {
    if (!value)
      return "";
    if (typeof value === "string")
      return value;
    if (typeof value === "object" && typeof value.mask === "string")
      return value.mask;
    return "";
  }

  function normalizeEvent(raw, index) {
    const tValue = Number(raw.t);
    return {
      t: Number.isFinite(tValue) ? tValue : index,
      event: raw.event ? String(raw.event) : "step",
      wave: Number.isFinite(Number(raw.wave)) ? Number(raw.wave) : 0,
      lane: Number.isFinite(Number(raw.lane)) ? Number(raw.lane) : 0,
      op: raw.op ? String(raw.op) : "",
      effect: raw.effect ? String(raw.effect) : "",
      blockSeq: Number.isFinite(Number(raw.blockSeq)) ? Number(raw.blockSeq) : null,
      blockKind: raw.blockKind ? String(raw.blockKind) : "",
      blockAddr: raw.blockAddr ? String(raw.blockAddr) : "",
      hasValue: typeof raw.hasValue === "boolean" ? raw.hasValue : undefined,
      active: normalizeMask(raw.active),
      expected: normalizeMask(raw.expected),
      index,
    };
  }

  function loadTraceFromText(text) {
    const lines = text.split(/\r?\n/);
    const parsed = [];
    for (const line of lines) {
      const entry = parseLine(line);
      if (entry)
        parsed.push(entry);
    }
    events = parsed.map((entry, index) => normalizeEvent(entry, index));
    currentIndex = -1;
    stopRun();
    updateView();
  }

  function trimMask(mask, lanes) {
    if (!mask)
      return "-";
    const match = mask.match(/0b([01]+)/);
    if (!match)
      return mask;
    const bits = match[1];
    const tail = bits.slice(-lanes);
    return `0b${tail}`;
  }

  function formatBlock(ev) {
    const kind = ev.blockKind || "block";
    if (Number.isFinite(ev.blockSeq))
      return `${kind}#${ev.blockSeq}`;
    if (ev.blockAddr)
      return ev.blockAddr;
    return "";
  }

  function updateStats() {
    eventCountEl.textContent = String(events.length);
    const position = currentIndex < 0 ? 0 : currentIndex + 1;
    eventIndexEl.textContent = String(position);
  }

  function buildLaneState() {
    const state = Array.from({ length: laneCount }, () => ({
      status: "idle",
      lastOp: "-",
      active: "",
      expected: "",
      blockLabel: "",
      lastEvent: null,
    }));

    for (let i = 0; i <= currentIndex && i < events.length; i++) {
      const ev = events[i];
      if (ev.lane < 0 || ev.lane >= laneCount)
        continue;
      const laneState = state[ev.lane];
      laneState.lastEvent = ev;
      laneState.active = ev.active;
      laneState.expected = ev.expected;
      laneState.lastOp = ev.op || ev.effect || ev.event;
      laneState.blockLabel = formatBlock(ev);
      if (ev.event === "suspend")
        laneState.status = "suspended";
      else if (ev.event === "return")
        laneState.status = "completed";
      else
        laneState.status = "running";
    }

    return state;
  }

  function renderLaneTable() {
    laneTable.innerHTML = "";
    if (!events.length) {
      laneTable.innerHTML = "<div class=\"hint\">Load a trace to see lanes.</div>";
      return;
    }

    const state = buildLaneState();
    for (let lane = 0; lane < laneCount; lane++) {
      const row = document.createElement("div");
      row.className = "lane-row";

      const laneLabel = document.createElement("div");
      laneLabel.textContent = `lane ${lane}`;

      const status = document.createElement("div");
      const laneState = state[lane];
      status.className = `lane-status ${laneState.status}`;
      let statusText =
        laneState.status === "idle" ? "idle" : `${laneState.status} - ${laneState.lastOp}`;
      if (laneState.blockLabel)
        statusText += ` @ ${laneState.blockLabel}`;
      status.textContent = statusText;

      const active = document.createElement("div");
      active.textContent = trimMask(laneState.active, laneCount);

      const expected = document.createElement("div");
      expected.textContent = trimMask(laneState.expected, laneCount);

      row.append(laneLabel, status, active, expected);
      laneTable.appendChild(row);
    }
  }

  function formatEventLine(ev) {
    const parts = [
      `#${ev.t}`,
      `wave ${ev.wave}`,
      `lane ${ev.lane}`,
      ev.event,
    ];
    if (ev.op)
      parts.push(ev.op);
    if (ev.effect)
      parts.push(`effect=${ev.effect}`);
    if (formatBlock(ev))
      parts.push(`block=${formatBlock(ev)}`);
    if (typeof ev.hasValue === "boolean")
      parts.push(`hasValue=${ev.hasValue}`);
    return parts.join(" - ");
  }

  function renderLog() {
    eventLog.innerHTML = "";
    if (!events.length) {
      eventLog.innerHTML = "<div class=\"log-entry\">Load a trace to see events.</div>";
      return;
    }
    if (currentIndex < 0) {
      eventLog.innerHTML = "<div class=\"log-entry\">Press Step or Run to start.</div>";
      return;
    }

    const start = Math.max(0, currentIndex - 40);
    const fragment = document.createDocumentFragment();
    for (let i = start; i <= currentIndex; i++) {
      const ev = events[i];
      const entry = document.createElement("div");
      entry.className = "log-entry";
      if (i === currentIndex)
        entry.classList.add("current");
      entry.textContent = formatEventLine(ev);
      fragment.appendChild(entry);
    }
    eventLog.appendChild(fragment);
    eventLog.scrollTop = eventLog.scrollHeight;
  }

  function isControlFlowOp(op) {
    if (!op)
      return false;
    return (
      op === "simt_step.if" ||
      op === "simt_step.loop" ||
      op === "simt_step.switch"
    );
  }

  function drawDiamond(ctx, x, y, size) {
    ctx.beginPath();
    ctx.moveTo(x, y - size);
    ctx.lineTo(x + size, y);
    ctx.lineTo(x, y + size);
    ctx.lineTo(x - size, y);
    ctx.closePath();
  }

  function resizeCanvas(ctx) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return rect;
  }

  function drawTimeline() {
    const ctx = canvas.getContext("2d");
    const rect = resizeCanvas(ctx);
    const width = rect.width;
    const height = rect.height;
    const pad = 24;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(9, 12, 16, 0.75)";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
    ctx.lineWidth = 1;
    for (let lane = 0; lane < laneCount; lane++) {
      const y = pad + (lane / Math.max(1, laneCount - 1)) * (height - pad * 2);
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(width - pad, y);
      ctx.stroke();
    }

    eventPoints = [];
    if (!events.length)
      return;

    const count = events.length;
    const xScale = count > 1 ? (width - pad * 2) / (count - 1) : 0;

    for (const ev of events) {
      if (ev.lane < 0 || ev.lane >= laneCount)
        continue;
      const x = pad + ev.index * xScale;
      const y = pad + (ev.lane / Math.max(1, laneCount - 1)) * (height - pad * 2);
      const isControl = ev.event === "step" && isControlFlowOp(ev.op);
      const color = isControl
        ? palette.control
        : (palette[ev.event] || palette.unknown);
      const active = ev.index <= currentIndex;
      ctx.globalAlpha = active ? 1 : 0.25;
      ctx.fillStyle = color;
      if (isControl) {
        drawDiamond(ctx, x, y, 5);
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.5)";
        ctx.lineWidth = 1;
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      eventPoints.push({ x, y, ev });
    }
    ctx.globalAlpha = 1;

    if (currentIndex >= 0 && currentIndex < events.length) {
      const x = pad + events[currentIndex].index * xScale;
      ctx.strokeStyle = "rgba(246, 176, 66, 0.7)";
      ctx.beginPath();
      ctx.moveTo(x, pad);
      ctx.lineTo(x, height - pad);
      ctx.stroke();
    }
  }

  function showTooltip(point) {
    const ev = point.ev;
    const parts = [
      `t=${ev.t}`,
      `wave=${ev.wave}`,
      `lane=${ev.lane}`,
      `event=${ev.event}`,
    ];
    if (ev.op)
      parts.push(`op=${ev.op}`);
    if (ev.effect)
      parts.push(`effect=${ev.effect}`);
    if (ev.blockKind)
      parts.push(`blockKind=${ev.blockKind}`);
    if (Number.isFinite(ev.blockSeq))
      parts.push(`blockSeq=${ev.blockSeq}`);
    if (ev.blockAddr)
      parts.push(`blockAddr=${ev.blockAddr}`);
    if (typeof ev.hasValue === "boolean")
      parts.push(`hasValue=${ev.hasValue}`);
    if (ev.active)
      parts.push(`active=${trimMask(ev.active, laneCount)}`);
    if (ev.expected)
      parts.push(`expected=${trimMask(ev.expected, laneCount)}`);
    tooltip.textContent = parts.join("\n");
    tooltip.style.left = `${point.x}px`;
    tooltip.style.top = `${point.y}px`;
    tooltip.classList.remove("hidden");
  }

  function hideTooltip() {
    tooltip.classList.add("hidden");
  }

  function updateView() {
    updateStats();
    drawTimeline();
    renderLaneTable();
    renderLog();
  }

  function stepForward() {
    if (currentIndex + 1 >= events.length)
      return false;
    currentIndex += 1;
    updateView();
    return true;
  }

  function stepBackward() {
    if (currentIndex <= -1)
      return false;
    currentIndex -= 1;
    updateView();
    return true;
  }

  function stopRun() {
    running = false;
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    runBtn.textContent = "Play";
  }

  function startRun() {
    if (running || !events.length)
      return;
    running = true;
    runBtn.textContent = "Pause";
    const interval = Math.max(1, Math.floor(1000 / speed));
    timer = setInterval(() => {
      if (!stepForward())
        stopRun();
    }, interval);
  }

  threadsInput.addEventListener("change", () => {
    laneCount = Math.max(1, Math.min(64, parseInt(threadsInput.value, 10) || 1));
    updateView();
  });

  speedInput.addEventListener("input", () => {
    speed = parseInt(speedInput.value, 10) || 1;
    speedLabel.textContent = String(speed);
    if (running) {
      stopRun();
      startRun();
    }
  });

  resetBtn.addEventListener("click", () => {
    currentIndex = -1;
    stopRun();
    updateView();
  });

  stepBtn.addEventListener("click", () => {
    stepForward();
  });

  backBtn.addEventListener("click", () => {
    if (running)
      stopRun();
    stepBackward();
  });

  runBtn.addEventListener("click", () => {
    if (running)
      stopRun();
    else
      startRun();
  });

  async function generateTrace() {
    const lanes = Math.max(1, Math.min(64, parseInt(threadsInput.value, 10) || 1));
    const seed = Math.max(0, parseInt(seedInput.value, 10) || 0);
    const program = programSelect.value || "richer";
    const query = new URLSearchParams({
      lanes: String(lanes),
      seed: String(seed),
      program,
    });
    if (collectiveCfInput && collectiveCfInput.checked)
      query.set("collective_cf", "1");

    generateBtn.disabled = true;
    runStatus.textContent = "Running interpreter...";
    runStatus.style.color = "var(--muted)";

    try {
      const response = await fetch(`/run?${query.toString()}`);
      const payload = await response.json();
      if (!response.ok || payload.error) {
        const message = payload.error || `Request failed (${response.status})`;
        throw new Error(message);
      }
      if (payload.trace) {
        loadTraceFromText(payload.trace);
        if (payload.ir)
          programText.value = payload.ir;
        runStatus.textContent = `Loaded ${events.length} events.`;
        runStatus.style.color = "var(--run)";
      } else {
        throw new Error("No trace data returned.");
      }
    } catch (err) {
      runStatus.textContent = err.message;
      runStatus.style.color = "var(--warn)";
    } finally {
      generateBtn.disabled = false;
    }
  }

  generateBtn.addEventListener("click", () => {
    generateTrace();
  });

  canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let closest = null;
    let minDist = 12;
    for (const point of eventPoints) {
      const dx = point.x - x;
      const dy = point.y - y;
      const dist = Math.hypot(dx, dy);
      if (dist < minDist) {
        minDist = dist;
        closest = point;
      }
    }
    if (closest)
      showTooltip(closest);
    else
      hideTooltip();
  });

  canvas.addEventListener("mouseleave", hideTooltip);
  window.addEventListener("resize", updateView);

  speedLabel.textContent = String(speed);
  updateView();
})();
