#include <cuda.h>
#include <nvrtc.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Dim3 {
    std::uint32_t x = 1;
    std::uint32_t y = 1;
    std::uint32_t z = 1;
};

enum class ScalarType {
    I32,
    U32,
    F32,
};

struct ScalarValue {
    ScalarType type = ScalarType::I32;
    int32_t i32 = 0;
    uint32_t u32 = 0;
    float f32 = 0.0f;
};

struct BufferDef {
    std::string name;
    ScalarType type = ScalarType::I32;
    std::size_t size = 0;
    bool hasFill = false;
    ScalarValue fill;
    std::unordered_map<std::size_t, ScalarValue> inits;
};

enum class BindingKind {
    Buffer,
    Const,
};

struct Binding {
    BindingKind kind = BindingKind::Buffer;
    std::string bufferName;
    ScalarType type = ScalarType::I32;
    ScalarValue value;
};

struct ExpectEntry {
    std::string buffer;
    std::size_t index = 0;
    ScalarValue expected;
    std::optional<float> absTol;
    std::optional<float> relTol;
    int line = 0;
};

struct ExpectRangeEntry {
    std::string buffer;
    std::size_t start = 0;
    std::size_t end = 0;
    ScalarValue expected;
    std::optional<float> absTol;
    std::optional<float> relTol;
    int line = 0;
};

struct Script {
    std::unordered_map<std::string, BufferDef> buffers;
    bool hasKernel = false;
    std::string kernelName = "main";
    std::string kernelSource;
    std::unordered_map<unsigned, Binding> bindings;
    bool hasLaunch = false;
    Dim3 grid;
    Dim3 block;
    std::vector<ExpectEntry> expects;
    std::vector<ExpectRangeEntry> expectRanges;
};

struct BufferRuntime {
    BufferDef def;
    CUdeviceptr device = 0;
    std::vector<int32_t> i32;
    std::vector<uint32_t> u32;
    std::vector<float> f32;
};

struct Options {
    std::string scriptPath;
    int deviceIndex = 0;
    std::string arch;
    bool dumpPtx = false;
};

static std::string trim(const std::string &value) {
    std::size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])))
        ++start;
    std::size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])))
        --end;
    return value.substr(start, end - start);
}

static bool isCommentLine(const std::string &line) {
    for (char ch : line) {
        if (!std::isspace(static_cast<unsigned char>(ch)))
            return ch == '#';
    }
    return false;
}

static std::vector<std::string> splitTokens(const std::string &line) {
    std::vector<std::string> tokens;
    std::istringstream stream(line);
    std::string tok;
    while (stream >> tok)
        tokens.push_back(tok);
    return tokens;
}

static void failLine(int line, const std::string &message) {
    std::cerr << "error: line " << line << ": " << message << "\n";
    std::exit(1);
}

static ScalarType parseType(const std::string &token, int line) {
    if (token == "i32")
        return ScalarType::I32;
    if (token == "u32")
        return ScalarType::U32;
    if (token == "f32")
        return ScalarType::F32;
    failLine(line, "unknown type '" + token + "'");
    return ScalarType::I32;
}

static std::uint64_t parseUnsigned(const std::string &token, int line) {
    try {
        std::size_t idx = 0;
        unsigned long long value = std::stoull(token, &idx, 0);
        if (idx != token.size())
            failLine(line, "invalid integer '" + token + "'");
        return static_cast<std::uint64_t>(value);
    } catch (const std::exception &) {
        failLine(line, "invalid integer '" + token + "'");
    }
    return 0;
}

static long long parseSigned(const std::string &token, int line) {
    try {
        std::size_t idx = 0;
        long long value = std::stoll(token, &idx, 0);
        if (idx != token.size())
            failLine(line, "invalid integer '" + token + "'");
        return value;
    } catch (const std::exception &) {
        failLine(line, "invalid integer '" + token + "'");
    }
    return 0;
}

static float parseFloat(const std::string &token, int line) {
    try {
        std::size_t idx = 0;
        double value = std::stod(token, &idx);
        if (idx != token.size())
            failLine(line, "invalid float '" + token + "'");
        return static_cast<float>(value);
    } catch (const std::exception &) {
        failLine(line, "invalid float '" + token + "'");
    }
    return 0.0f;
}

static ScalarValue parseValue(const std::string &token, ScalarType type, int line) {
    ScalarValue value;
    value.type = type;
    switch (type) {
        case ScalarType::I32: {
            long long parsed = parseSigned(token, line);
            if (parsed < std::numeric_limits<int32_t>::min() ||
                parsed > std::numeric_limits<int32_t>::max()) {
                failLine(line, "i32 out of range: " + token);
            }
            value.i32 = static_cast<int32_t>(parsed);
            break;
        }
        case ScalarType::U32: {
            unsigned long long parsed = parseUnsigned(token, line);
            if (parsed > std::numeric_limits<uint32_t>::max())
                failLine(line, "u32 out of range: " + token);
            value.u32 = static_cast<uint32_t>(parsed);
            break;
        }
        case ScalarType::F32:
            value.f32 = parseFloat(token, line);
            break;
    }
    return value;
}

static void checkCuda(CUresult result, const char *what) {
    if (result == CUDA_SUCCESS)
        return;
    const char *name = nullptr;
    const char *desc = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &desc);
    std::cerr << "CUDA error: " << what;
    if (name)
        std::cerr << " (" << name << ")";
    if (desc)
        std::cerr << ": " << desc;
    std::cerr << "\n";
    std::exit(1);
}

static void checkNvrtc(nvrtcResult result, const char *what, nvrtcProgram program) {
    if (result == NVRTC_SUCCESS)
        return;
    std::cerr << "NVRTC error: " << what << ": " << nvrtcGetErrorString(result) << "\n";
    size_t logSize = 0;
    if (nvrtcGetProgramLogSize(program, &logSize) == NVRTC_SUCCESS && logSize > 1) {
        std::string log(logSize, '\0');
        if (nvrtcGetProgramLog(program, log.data()) == NVRTC_SUCCESS)
            std::cerr << log << "\n";
    }
    std::exit(1);
}

static void parseScript(const std::string &path, Script &script) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "error: failed to open script '" << path << "'\n";
        std::exit(1);
    }

    std::string line;
    int lineNumber = 0;
    bool inKernel = false;
    while (std::getline(file, line)) {
        ++lineNumber;
        if (inKernel) {
            std::string trimmed = trim(line);
            if (trimmed == "ENDKERNEL") {
                inKernel = false;
                continue;
            }
            script.kernelSource.append(line);
            script.kernelSource.push_back('\n');
            continue;
        }

        if (isCommentLine(line))
            continue;
        std::string trimmed = trim(line);
        if (trimmed.empty())
            continue;

        std::vector<std::string> tokens = splitTokens(trimmed);
        const std::string &cmd = tokens[0];

        if (cmd == "BUFFER") {
            if (tokens.size() != 6 || tokens[2] != "TYPE" || tokens[4] != "SIZE")
                failLine(lineNumber, "BUFFER syntax: BUFFER <name> TYPE <type> SIZE <n>");
            const std::string &name = tokens[1];
            if (script.buffers.count(name))
                failLine(lineNumber, "BUFFER already defined: " + name);
            BufferDef def;
            def.name = name;
            def.type = parseType(tokens[3], lineNumber);
            def.size = parseUnsigned(tokens[5], lineNumber);
            if (def.size == 0)
                failLine(lineNumber, "BUFFER size must be > 0");
            script.buffers[name] = def;
            continue;
        }

        if (cmd == "FILL") {
            if (tokens.size() != 3)
                failLine(lineNumber, "FILL syntax: FILL <name> <value>");
            auto it = script.buffers.find(tokens[1]);
            if (it == script.buffers.end())
                failLine(lineNumber, "unknown buffer: " + tokens[1]);
            BufferDef &def = it->second;
            if (def.hasFill)
                failLine(lineNumber, "FILL already set for buffer: " + def.name);
            def.fill = parseValue(tokens[2], def.type, lineNumber);
            def.hasFill = true;
            continue;
        }

        if (cmd == "INIT") {
            if (tokens.size() != 4)
                failLine(lineNumber, "INIT syntax: INIT <name> <index> <value>");
            auto it = script.buffers.find(tokens[1]);
            if (it == script.buffers.end())
                failLine(lineNumber, "unknown buffer: " + tokens[1]);
            BufferDef &def = it->second;
            std::size_t index = parseUnsigned(tokens[2], lineNumber);
            if (index >= def.size)
                failLine(lineNumber, "INIT index out of range");
            if (def.inits.count(index))
                failLine(lineNumber, "INIT already set for buffer index");
            def.inits[index] = parseValue(tokens[3], def.type, lineNumber);
            continue;
        }

        if (cmd == "KERNEL") {
            if (script.hasKernel)
                failLine(lineNumber, "KERNEL already defined");
            if (tokens.size() != 1 && tokens.size() != 2)
                failLine(lineNumber, "KERNEL syntax: KERNEL [<name>]");
            if (tokens.size() == 2)
                script.kernelName = tokens[1];
            script.hasKernel = true;
            inKernel = true;
            continue;
        }

        if (cmd == "BIND") {
            if (tokens.size() >= 2 && tokens[1] == "CONST") {
                if (tokens.size() != 7 || tokens[3] != "TYPE" || tokens[5] != "ARG")
                    failLine(lineNumber,
                             "BIND CONST syntax: BIND CONST <value> TYPE <type> ARG <n>");
                Binding binding;
                binding.kind = BindingKind::Const;
                binding.type = parseType(tokens[4], lineNumber);
                binding.value = parseValue(tokens[2], binding.type, lineNumber);
                unsigned argIndex = static_cast<unsigned>(parseUnsigned(tokens[6], lineNumber));
                if (script.bindings.count(argIndex))
                    failLine(lineNumber, "duplicate BIND ARG index");
                script.bindings[argIndex] = binding;
            } else {
                if (tokens.size() != 4 || tokens[2] != "ARG")
                    failLine(lineNumber, "BIND syntax: BIND <name> ARG <n>");
                const std::string &name = tokens[1];
                if (!script.buffers.count(name))
                    failLine(lineNumber, "unknown buffer: " + name);
                Binding binding;
                binding.kind = BindingKind::Buffer;
                binding.bufferName = name;
                binding.type = script.buffers.at(name).type;
                unsigned argIndex = static_cast<unsigned>(parseUnsigned(tokens[3], lineNumber));
                if (script.bindings.count(argIndex))
                    failLine(lineNumber, "duplicate BIND ARG index");
                script.bindings[argIndex] = binding;
            }
            continue;
        }

        if (cmd == "LAUNCH") {
            if (script.hasLaunch)
                failLine(lineNumber, "LAUNCH already set");
            if (tokens.size() != 9 || tokens[1] != "GRID" || tokens[5] != "BLOCK") {
                failLine(lineNumber,
                         "LAUNCH syntax: LAUNCH GRID <x> <y> <z> BLOCK <x> <y> <z>");
            }
            script.grid.x = static_cast<std::uint32_t>(parseUnsigned(tokens[2], lineNumber));
            script.grid.y = static_cast<std::uint32_t>(parseUnsigned(tokens[3], lineNumber));
            script.grid.z = static_cast<std::uint32_t>(parseUnsigned(tokens[4], lineNumber));
            script.block.x = static_cast<std::uint32_t>(parseUnsigned(tokens[6], lineNumber));
            script.block.y = static_cast<std::uint32_t>(parseUnsigned(tokens[7], lineNumber));
            script.block.z = static_cast<std::uint32_t>(parseUnsigned(tokens[8], lineNumber));
            if (script.grid.x == 0 || script.grid.y == 0 || script.grid.z == 0 ||
                script.block.x == 0 || script.block.y == 0 || script.block.z == 0) {
                failLine(lineNumber, "LAUNCH dimensions must be > 0");
            }
            script.hasLaunch = true;
            continue;
        }

        if (cmd == "EXPECT") {
            if (tokens.size() < 4)
                failLine(lineNumber, "EXPECT syntax: EXPECT <name> <index> <value> ...");
            const std::string &bufferName = tokens[1];
            auto it = script.buffers.find(bufferName);
            if (it == script.buffers.end())
                failLine(lineNumber, "unknown buffer: " + bufferName);
            ExpectEntry entry;
            entry.buffer = bufferName;
            entry.index = parseUnsigned(tokens[2], lineNumber);
            if (entry.index >= it->second.size)
                failLine(lineNumber, "EXPECT index out of range");
            entry.expected = parseValue(tokens[3], it->second.type, lineNumber);
            entry.line = lineNumber;
            for (std::size_t i = 4; i < tokens.size();) {
                if (tokens[i] == "ABS_TOL") {
                    if (i + 1 >= tokens.size())
                        failLine(lineNumber, "ABS_TOL missing value");
                    entry.absTol = parseFloat(tokens[i + 1], lineNumber);
                    i += 2;
                    continue;
                }
                if (tokens[i] == "REL_TOL") {
                    if (i + 1 >= tokens.size())
                        failLine(lineNumber, "REL_TOL missing value");
                    entry.relTol = parseFloat(tokens[i + 1], lineNumber);
                    i += 2;
                    continue;
                }
                failLine(lineNumber, "unknown EXPECT modifier: " + tokens[i]);
            }
            script.expects.push_back(entry);
            continue;
        }

        if (cmd == "EXPECT_RANGE") {
            if (tokens.size() < 5)
                failLine(lineNumber,
                         "EXPECT_RANGE syntax: EXPECT_RANGE <name> <start> <end> <value> ...");
            const std::string &bufferName = tokens[1];
            auto it = script.buffers.find(bufferName);
            if (it == script.buffers.end())
                failLine(lineNumber, "unknown buffer: " + bufferName);
            ExpectRangeEntry entry;
            entry.buffer = bufferName;
            entry.start = parseUnsigned(tokens[2], lineNumber);
            entry.end = parseUnsigned(tokens[3], lineNumber);
            if (entry.start > entry.end)
                failLine(lineNumber, "EXPECT_RANGE start > end");
            if (entry.end >= it->second.size)
                failLine(lineNumber, "EXPECT_RANGE end out of range");
            entry.expected = parseValue(tokens[4], it->second.type, lineNumber);
            entry.line = lineNumber;
            for (std::size_t i = 5; i < tokens.size();) {
                if (tokens[i] == "ABS_TOL") {
                    if (i + 1 >= tokens.size())
                        failLine(lineNumber, "ABS_TOL missing value");
                    entry.absTol = parseFloat(tokens[i + 1], lineNumber);
                    i += 2;
                    continue;
                }
                if (tokens[i] == "REL_TOL") {
                    if (i + 1 >= tokens.size())
                        failLine(lineNumber, "REL_TOL missing value");
                    entry.relTol = parseFloat(tokens[i + 1], lineNumber);
                    i += 2;
                    continue;
                }
                failLine(lineNumber, "unknown EXPECT_RANGE modifier: " + tokens[i]);
            }
            script.expectRanges.push_back(entry);
            continue;
        }

        failLine(lineNumber, "unknown command: " + cmd);
    }

    if (inKernel)
        failLine(lineNumber, "missing ENDKERNEL");
    if (!script.hasKernel)
        failLine(lineNumber, "missing KERNEL block");
    if (script.kernelSource.empty())
        failLine(lineNumber, "empty kernel source");
    if (!script.hasLaunch)
        failLine(lineNumber, "missing LAUNCH command");
    if (!script.bindings.empty()) {
        unsigned maxArg = 0;
        for (const auto &entry : script.bindings)
            maxArg = std::max(maxArg, entry.first);
        for (unsigned i = 0; i <= maxArg; ++i) {
            if (!script.bindings.count(i))
                failLine(lineNumber, "missing BIND for ARG " + std::to_string(i));
        }
    }
}

static void applyFill(BufferRuntime &runtime) {
    switch (runtime.def.type) {
        case ScalarType::I32:
            runtime.i32.assign(runtime.def.size, runtime.def.hasFill ? runtime.def.fill.i32 : 0);
            break;
        case ScalarType::U32:
            runtime.u32.assign(runtime.def.size, runtime.def.hasFill ? runtime.def.fill.u32 : 0u);
            break;
        case ScalarType::F32:
            runtime.f32.assign(runtime.def.size, runtime.def.hasFill ? runtime.def.fill.f32 : 0.0f);
            break;
    }
}

static void applyInits(BufferRuntime &runtime) {
    for (const auto &entry : runtime.def.inits) {
        std::size_t index = entry.first;
        const ScalarValue &value = entry.second;
        switch (runtime.def.type) {
            case ScalarType::I32:
                runtime.i32[index] = value.i32;
                break;
            case ScalarType::U32:
                runtime.u32[index] = value.u32;
                break;
            case ScalarType::F32:
                runtime.f32[index] = value.f32;
                break;
        }
    }
}

static void copyHostToDevice(const BufferRuntime &runtime) {
    std::size_t bytes = 0;
    const void *data = nullptr;
    switch (runtime.def.type) {
        case ScalarType::I32:
            bytes = runtime.i32.size() * sizeof(int32_t);
            data = runtime.i32.data();
            break;
        case ScalarType::U32:
            bytes = runtime.u32.size() * sizeof(uint32_t);
            data = runtime.u32.data();
            break;
        case ScalarType::F32:
            bytes = runtime.f32.size() * sizeof(float);
            data = runtime.f32.data();
            break;
    }
    checkCuda(cuMemcpyHtoD(runtime.device, data, bytes), "cuMemcpyHtoD");
}

static void copyDeviceToHost(BufferRuntime &runtime) {
    std::size_t bytes = 0;
    void *data = nullptr;
    switch (runtime.def.type) {
        case ScalarType::I32:
            bytes = runtime.i32.size() * sizeof(int32_t);
            data = runtime.i32.data();
            break;
        case ScalarType::U32:
            bytes = runtime.u32.size() * sizeof(uint32_t);
            data = runtime.u32.data();
            break;
        case ScalarType::F32:
            bytes = runtime.f32.size() * sizeof(float);
            data = runtime.f32.data();
            break;
    }
    checkCuda(cuMemcpyDtoH(data, runtime.device, bytes), "cuMemcpyDtoH");
}

static bool checkFloat(float actual, float expected, float absTol, float relTol) {
    double diff = std::abs(static_cast<double>(actual) - static_cast<double>(expected));
    double tol = absTol;
    tol = std::max(tol, relTol * std::abs(static_cast<double>(expected)));
    return diff <= tol;
}

static void verifyExpect(const BufferRuntime &runtime, const ExpectEntry &expect) {
    bool ok = false;
    switch (runtime.def.type) {
        case ScalarType::I32:
            ok = (runtime.i32[expect.index] == expect.expected.i32);
            if (expect.absTol || expect.relTol)
                failLine(expect.line, "tolerances are not allowed for integer EXPECT");
            break;
        case ScalarType::U32:
            ok = (runtime.u32[expect.index] == expect.expected.u32);
            if (expect.absTol || expect.relTol)
                failLine(expect.line, "tolerances are not allowed for integer EXPECT");
            break;
        case ScalarType::F32: {
            float absTol = expect.absTol.value_or(0.0f);
            float relTol = expect.relTol.value_or(0.0f);
            ok = checkFloat(runtime.f32[expect.index], expect.expected.f32, absTol, relTol);
            break;
        }
    }
    if (!ok) {
        std::cerr << "EXPECT failed: buffer=" << runtime.def.name
                  << " index=" << expect.index << "\n";
        std::exit(1);
    }
}

static void verifyExpectRange(const BufferRuntime &runtime, const ExpectRangeEntry &expect) {
    if (runtime.def.type != ScalarType::F32 && (expect.absTol || expect.relTol))
        failLine(expect.line, "tolerances are not allowed for integer EXPECT_RANGE");
    if (runtime.def.type == ScalarType::F32) {
        if ((expect.absTol && *expect.absTol < 0.0f) || (expect.relTol && *expect.relTol < 0.0f))
            failLine(expect.line, "invalid float tolerance");
    }

    float absTol = expect.absTol.value_or(0.0f);
    float relTol = expect.relTol.value_or(0.0f);

    for (std::size_t i = expect.start; i <= expect.end; ++i) {
        bool ok = false;
        switch (runtime.def.type) {
            case ScalarType::I32:
                ok = (runtime.i32[i] == expect.expected.i32);
                break;
            case ScalarType::U32:
                ok = (runtime.u32[i] == expect.expected.u32);
                break;
            case ScalarType::F32:
                ok = checkFloat(runtime.f32[i], expect.expected.f32, absTol, relTol);
                break;
        }
        if (!ok) {
            std::cerr << "EXPECT_RANGE failed: buffer=" << runtime.def.name
                      << " index=" << i << "\n";
            std::exit(1);
        }
    }
}

static void runScript(const Script &script, const Options &options) {
    checkCuda(cuInit(0), "cuInit");
    CUdevice device = 0;
    checkCuda(cuDeviceGet(&device, options.deviceIndex), "cuDeviceGet");
    CUcontext context = nullptr;
    checkCuda(cuCtxCreate(&context, 0, device), "cuCtxCreate");

    nvrtcProgram program;
    checkNvrtc(nvrtcCreateProgram(&program,
                                  script.kernelSource.c_str(),
                                  "script.cu",
                                  0,
                                  nullptr,
                                  nullptr),
               "nvrtcCreateProgram", program);

    std::vector<std::string> optStorage;
    optStorage.push_back("--std=c++17");
    if (!options.arch.empty())
        optStorage.push_back("--gpu-architecture=" + options.arch);

    std::vector<const char *> optPtrs;
    optPtrs.reserve(optStorage.size());
    for (const auto &opt : optStorage)
        optPtrs.push_back(opt.c_str());

    nvrtcResult compileResult = nvrtcCompileProgram(program,
                                                    static_cast<int>(optPtrs.size()),
                                                    optPtrs.data());
    if (compileResult != NVRTC_SUCCESS)
        checkNvrtc(compileResult, "nvrtcCompileProgram", program);

    size_t ptxSize = 0;
    checkNvrtc(nvrtcGetPTXSize(program, &ptxSize), "nvrtcGetPTXSize", program);
    std::string ptx(ptxSize, '\0');
    checkNvrtc(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX", program);
    checkNvrtc(nvrtcDestroyProgram(&program), "nvrtcDestroyProgram", program);

    if (options.dumpPtx)
        std::cout << ptx << "\n";

    CUmodule module = nullptr;
    checkCuda(cuModuleLoadDataEx(&module, ptx.data(), 0, nullptr, nullptr), "cuModuleLoadDataEx");

    CUfunction kernel = nullptr;
    checkCuda(cuModuleGetFunction(&kernel, module, script.kernelName.c_str()), "cuModuleGetFunction");

    std::unordered_map<std::string, BufferRuntime> runtimes;
    for (const auto &entry : script.buffers) {
        BufferRuntime runtime;
        runtime.def = entry.second;
        applyFill(runtime);
        applyInits(runtime);
        std::size_t bytes = 0;
        switch (runtime.def.type) {
            case ScalarType::I32:
                bytes = runtime.i32.size() * sizeof(int32_t);
                break;
            case ScalarType::U32:
                bytes = runtime.u32.size() * sizeof(uint32_t);
                break;
            case ScalarType::F32:
                bytes = runtime.f32.size() * sizeof(float);
                break;
        }
        checkCuda(cuMemAlloc(&runtime.device, bytes), "cuMemAlloc");
        copyHostToDevice(runtime);
        runtimes[entry.first] = std::move(runtime);
    }

    std::vector<std::vector<std::uint8_t>> argStorage;
    std::vector<void *> argPtrs;
    if (!script.bindings.empty()) {
        unsigned maxArg = 0;
        for (const auto &entry : script.bindings)
            maxArg = std::max(maxArg, entry.first);
        argStorage.reserve(maxArg + 1);
        argPtrs.reserve(maxArg + 1);

        for (unsigned i = 0; i <= maxArg; ++i) {
            const Binding &binding = script.bindings.at(i);
            if (binding.kind == BindingKind::Buffer) {
                const auto &runtime = runtimes.at(binding.bufferName);
                CUdeviceptr ptr = runtime.device;
                std::vector<std::uint8_t> data(sizeof(CUdeviceptr));
                std::memcpy(data.data(), &ptr, sizeof(CUdeviceptr));
                argStorage.push_back(std::move(data));
            } else {
                std::vector<std::uint8_t> data;
                switch (binding.type) {
                    case ScalarType::I32: {
                        int32_t value = binding.value.i32;
                        data.resize(sizeof(value));
                        std::memcpy(data.data(), &value, sizeof(value));
                        break;
                    }
                    case ScalarType::U32: {
                        uint32_t value = binding.value.u32;
                        data.resize(sizeof(value));
                        std::memcpy(data.data(), &value, sizeof(value));
                        break;
                    }
                    case ScalarType::F32: {
                        float value = binding.value.f32;
                        data.resize(sizeof(value));
                        std::memcpy(data.data(), &value, sizeof(value));
                        break;
                    }
                }
                argStorage.push_back(std::move(data));
            }
            argPtrs.push_back(argStorage.back().data());
        }
    }

    checkCuda(cuLaunchKernel(kernel,
                             script.grid.x,
                             script.grid.y,
                             script.grid.z,
                             script.block.x,
                             script.block.y,
                             script.block.z,
                             0,
                             nullptr,
                             argPtrs.empty() ? nullptr : argPtrs.data(),
                             nullptr),
              "cuLaunchKernel");
    checkCuda(cuCtxSynchronize(), "cuCtxSynchronize");

    for (auto &entry : runtimes)
        copyDeviceToHost(entry.second);

    for (const auto &expect : script.expects) {
        const auto &runtime = runtimes.at(expect.buffer);
        verifyExpect(runtime, expect);
    }

    for (const auto &expect : script.expectRanges) {
        const auto &runtime = runtimes.at(expect.buffer);
        verifyExpectRange(runtime, expect);
    }

    for (auto &entry : runtimes)
        checkCuda(cuMemFree(entry.second.device), "cuMemFree");

    checkCuda(cuModuleUnload(module), "cuModuleUnload");
    checkCuda(cuCtxDestroy(context), "cuCtxDestroy");
}

static Options parseOptions(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device") {
            if (i + 1 >= argc) {
                std::cerr << "error: --device requires a value\n";
                std::exit(1);
            }
            try {
                options.deviceIndex = std::stoi(argv[++i]);
            } catch (const std::exception &) {
                std::cerr << "error: invalid --device value\n";
                std::exit(1);
            }
            if (options.deviceIndex < 0) {
                std::cerr << "error: --device must be >= 0\n";
                std::exit(1);
            }
            continue;
        }
        if (arg == "--arch") {
            if (i + 1 >= argc) {
                std::cerr << "error: --arch requires a value\n";
                std::exit(1);
            }
            options.arch = argv[++i];
            continue;
        }
        if (arg == "--dump-ptx") {
            options.dumpPtx = true;
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            std::cerr << "error: unknown option '" << arg << "'\n";
            std::exit(1);
        }
        if (!options.scriptPath.empty()) {
            std::cerr << "error: multiple script paths provided\n";
            std::exit(1);
        }
        options.scriptPath = arg;
    }

    if (options.scriptPath.empty()) {
        std::cerr << "usage: simt-cuda-test <script.cuda> [--device N] [--arch sm_80] [--dump-ptx]\n";
        std::exit(1);
    }

    return options;
}

} // namespace

int main(int argc, char **argv) {
    Options options = parseOptions(argc, argv);

    Script script;
    parseScript(options.scriptPath, script);
    runScript(script, options);

    return 0;
}
