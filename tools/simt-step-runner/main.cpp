#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/semantics/SimpleProgram.h"
#include "simt-step/semantics/TraceJsonWriter.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/YAMLTraits.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <optional>
#include <vector>

using namespace mlir;

namespace {
struct InitSpec {
    unsigned argIndex = 0;
    int64_t index = 0;
    int64_t value = 0;
};

struct InitEntry {
    int64_t index = 0;
    int64_t value = 0;
};

struct InitBuffer {
    std::string buffer;
    std::optional<int64_t> size;
    std::optional<int64_t> fill;
    std::vector<InitEntry> entries;
};

struct InitFile {
    std::vector<InitBuffer> buffers;
};

} // namespace

LLVM_YAML_IS_SEQUENCE_VECTOR(InitEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(InitBuffer)

namespace llvm::yaml {

template <>
struct MappingTraits<InitEntry> {
    static void mapping(IO &io, InitEntry &entry) {
        io.mapRequired("index", entry.index);
        io.mapRequired("value", entry.value);
    }
};

template <>
struct MappingTraits<InitBuffer> {
    static void mapping(IO &io, InitBuffer &buffer) {
        io.mapRequired("buffer", buffer.buffer);
        io.mapOptional("size", buffer.size);
        io.mapOptional("fill", buffer.fill);
        io.mapOptional("entries", buffer.entries);
    }
};

template <>
struct MappingTraits<InitFile> {
    static void mapping(IO &io, InitFile &file) {
        io.mapOptional("buffers", file.buffers);
    }
};

} // namespace llvm::yaml

namespace {

static bool parseInt64(llvm::StringRef input, int64_t &out) {
    if (input.empty())
        return false;
    std::string text = input.str();
    char *end = nullptr;
    errno = 0;
    long long value = std::strtoll(text.c_str(), &end, 0);
    if (errno != 0 || end != text.c_str() + text.size())
        return false;
    out = static_cast<int64_t>(value);
    return true;
}

static bool parseBufferRef(llvm::StringRef name, unsigned &argIndex,
                           std::string &error) {
    llvm::StringRef rest = name;
    if (!(rest.consume_front("buf") || rest.consume_front("arg"))) {
        error = "buffer must start with bufN or argN";
        return false;
    }
    int64_t idx = 0;
    if (!parseInt64(rest, idx) || idx < 0) {
        error = "invalid buffer argument index";
        return false;
    }
    argIndex = static_cast<unsigned>(idx);
    return true;
}

static bool parseInitSpec(llvm::StringRef spec, InitSpec &out,
                          std::string &error) {
    llvm::SmallVector<llvm::StringRef, 4> parts;
    spec.split(parts, ':');
    if (parts.size() != 3) {
        error = "expected format bufN:idx:value";
        return false;
    }
    if (!parseBufferRef(parts[0], out.argIndex, error)) {
        return false;
    }
    if (!parseInt64(parts[1], out.index)) {
        error = "invalid buffer index";
        return false;
    }
    if (!parseInt64(parts[2], out.value)) {
        error = "invalid buffer value";
        return false;
    }
    return true;
}

static bool loadInitFile(llvm::StringRef path, InitFile &out,
                         std::string &error) {
    auto fileOrErr = llvm::MemoryBuffer::getFile(path);
    if (!fileOrErr) {
        error = fileOrErr.getError().message();
        return false;
    }
    llvm::yaml::Input yin(fileOrErr.get()->getBuffer());
    yin >> out;
    if (auto err = yin.error()) {
        error = err.message();
        return false;
    }
    return true;
}

static simt::semantics::SemValue castInitValue(mlir::Type elementType,
                                               int64_t value) {
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
        if (intTy.getWidth() <= 32)
            return simt::semantics::SemValue::fromInt32(
                static_cast<int32_t>(value));
        return simt::semantics::SemValue::fromInt64(value);
    }
    if (mlir::isa<mlir::IndexType>(elementType))
        return simt::semantics::SemValue::fromInt64(value);
    if (mlir::isa<mlir::FloatType>(elementType))
        return simt::semantics::SemValue::fromFloat(static_cast<float>(value));
    llvm::report_fatal_error("unsupported buffer element type for --init");
}
} // namespace

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);
    llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                         llvm::cl::desc("<input mlir>"),
                                         llvm::cl::init("-"));
    llvm::cl::opt<std::string> funcName(
        "func", llvm::cl::desc("Entry function name"),
        llvm::cl::init("main"));
    llvm::cl::opt<unsigned> numLanes(
        "lanes", llvm::cl::desc("Number of lanes to execute"),
        llvm::cl::init(4));
    llvm::cl::opt<unsigned> subgroupWidth(
        "subgroup-width", llvm::cl::desc("Subgroup width"),
        llvm::cl::init(8));
    llvm::cl::opt<bool> dumpIR(
        "print-ir", llvm::cl::desc("Print parsed IR before running"),
        llvm::cl::init(false));
    llvm::cl::opt<bool> collectiveControlFlow(
        "collective-cf",
        llvm::cl::desc("Make control-flow ops collective before split"),
        llvm::cl::init(false));
    llvm::cl::opt<bool> syncControlFlow(
        "sync-cf",
        llvm::cl::desc("Make control-flow ops synchronous (barriered)"),
        llvm::cl::init(false));
    llvm::cl::opt<bool> syncMemory(
        "sync-mem",
        llvm::cl::desc("Make buffer load/store synchronous (barriered)"),
        llvm::cl::init(false));
    llvm::cl::opt<bool> collectiveMemory(
        "collective-mem",
        llvm::cl::desc("Make buffer load/store collective"),
        llvm::cl::init(false));
    llvm::cl::opt<std::string> traceFile(
        "trace-file", llvm::cl::desc("Write interpreter trace to JSONL file"),
        llvm::cl::init(""));
    llvm::cl::opt<std::string> initFile(
        "init-file",
        llvm::cl::desc("YAML file for buffer initialization"),
        llvm::cl::init(""));
    llvm::cl::list<std::string> initBuffers(
        "init",
        llvm::cl::desc("Initialize buffer entries as bufN:idx:value"),
        llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);
    llvm::cl::ParseCommandLineOptions(argc, argv, "simt-step runner\n");

    DialectRegistry registry;
    simt::dialect::registerSimtStepDialect(registry);
    registry.insert<arith::ArithDialect, func::FuncDialect>();
    MLIRContext context(registry);
    context.loadAllAvailableDialects();
    (void)context.getOrLoadDialect<simt::dialect::SimtStepDialect>();
    (void)context.getOrLoadDialect<arith::ArithDialect>();
    (void)context.getOrLoadDialect<func::FuncDialect>();

    auto module = parseSourceFile<ModuleOp>(inputFile, &context);
    if (!module) {
        llvm::errs() << "failed to parse module\n";
        return 1;
    }

    if (dumpIR) {
        module->print(llvm::outs());
        llvm::outs() << "\n";
    }

    auto func = module->lookupSymbol<func::FuncOp>(funcName);
    if (!func) {
        llvm::errs() << "module missing @" << funcName << "\n";
        return 1;
    }

    if (collectiveControlFlow && syncControlFlow) {
        llvm::errs() << "error: --collective-cf conflicts with --sync-cf\n";
        return 1;
    }
    if (collectiveMemory && syncMemory) {
        llvm::errs() << "error: --collective-mem conflicts with --sync-mem\n";
        return 1;
    }

    simt::semantics::ExecutionPolicy execPolicy;
    if (collectiveControlFlow)
        execPolicy.controlFlow = simt::semantics::ExecutionMode::Collective;
    else if (syncControlFlow)
        execPolicy.controlFlow = simt::semantics::ExecutionMode::Synchronous;
    if (collectiveMemory)
        execPolicy.memoryOps = simt::semantics::ExecutionMode::Collective;
    else if (syncMemory)
        execPolicy.memoryOps = simt::semantics::ExecutionMode::Synchronous;

    simt::semantics::SemanticsContext semaCtx;
    unsigned width = std::min<unsigned>(64, std::max<unsigned>(1, numLanes));
    semaCtx.activeMask =
        width >= 64 ? ~0ull : ((1ull << static_cast<std::uint64_t>(width)) - 1ull);
    semaCtx.subgroupWidth = std::max<unsigned>(1, subgroupWidth);
    semaCtx.policy = &execPolicy;

    simt::semantics::SimpleSemantics::clearMemory();
    auto &memMutable = simt::semantics::SimpleSemantics::memoryMutable();
    if (!initFile.empty()) {
        InitFile file;
        std::string error;
        if (!loadInitFile(initFile, file, error)) {
            llvm::errs() << "failed to parse --init-file: " << error << "\n";
            return 1;
        }
        for (const auto &buffer : file.buffers) {
            InitSpec init;
            if (!parseBufferRef(buffer.buffer, init.argIndex, error)) {
                llvm::errs() << "invalid --init-file buffer '" << buffer.buffer
                             << "': " << error << "\n";
                return 1;
            }
            if (init.argIndex >= func.getNumArguments()) {
                llvm::errs() << "invalid --init-file buffer '" << buffer.buffer
                             << "': argument out of range\n";
                return 1;
            }
            mlir::Value arg = func.getArgument(init.argIndex);
            auto resTy = mlir::dyn_cast<simt::dialect::ResourceType>(arg.getType());
            if (!resTy) {
                llvm::errs() << "invalid --init-file buffer '" << buffer.buffer
                             << "': argument is not a resource\n";
                return 1;
            }
            if (buffer.fill) {
                if (!buffer.size) {
                    llvm::errs() << "invalid --init-file buffer '"
                                 << buffer.buffer << "': fill requires size\n";
                    return 1;
                }
                if (*buffer.size < 0) {
                    llvm::errs() << "invalid --init-file buffer '"
                                 << buffer.buffer << "': size must be >= 0\n";
                    return 1;
                }
                auto fillValue =
                    castInitValue(resTy.getElementType(), *buffer.fill);
                for (int64_t i = 0; i < *buffer.size; ++i)
                    memMutable[arg][i] = fillValue;
            }
            for (const auto &entry : buffer.entries) {
                memMutable[arg][entry.index] =
                    castInitValue(resTy.getElementType(), entry.value);
            }
        }
    }
    if (!initBuffers.empty()) {
        for (const auto &spec : initBuffers) {
            InitSpec init;
            std::string error;
            if (!parseInitSpec(spec, init, error)) {
                llvm::errs() << "invalid --init '" << spec << "': " << error << "\n";
                return 1;
            }
            if (init.argIndex >= func.getNumArguments()) {
                llvm::errs() << "invalid --init '" << spec
                             << "': argument out of range\n";
                return 1;
            }
            mlir::Value arg = func.getArgument(init.argIndex);
            auto resTy = mlir::dyn_cast<simt::dialect::ResourceType>(arg.getType());
            if (!resTy) {
                llvm::errs() << "invalid --init '" << spec
                             << "': argument is not a resource\n";
                return 1;
            }
            memMutable[arg][init.index] =
                castInitValue(resTy.getElementType(), init.value);
        }
    }
    simt::semantics::SimpleProgramRunner runner;
    std::unique_ptr<simt::semantics::TraceJsonWriter> traceWriter;
    if (!traceFile.empty()) {
        traceWriter = std::make_unique<simt::semantics::TraceJsonWriter>(traceFile);
        runner.setTraceSink(traceWriter.get());
    }

    auto &entry = func.getBody().front();
    if (llvm::Error err = runner.runBlock(&entry, semaCtx)) {
        llvm::errs() << "run failed: " << llvm::toString(std::move(err)) << "\n";
        return 1;
    }

    const auto &state = runner.state();
    for (const auto &waveIt : state.waves) {
        llvm::outs() << "Wave " << waveIt.first << "\n";
        const auto &waveCtx = waveIt.second;
        for (const auto &laneIt : waveCtx.lanes) {
            const auto &laneCtx = laneIt.second;
            llvm::outs() << "  Lane " << laneIt.first
                         << " returned=" << laneCtx.hasReturned;
            if (laneCtx.returnValue)
                llvm::outs() << " value=" << laneCtx.returnValue->asInt64();
            if (laneCtx.currentBlock)
                llvm::outs() << " block=" << laneCtx.currentBlock->block
                             << " seq=" << laneCtx.currentBlock->sequenceId;
            llvm::outs() << "\n";
        }
    }

    const auto &mem = simt::semantics::SimpleSemantics::memory();
    if (!mem.empty()) {
        llvm::outs() << "Memory:\n";
        for (const auto &resIt : mem) {
            std::string bufName = "res";
            if (auto barg = mlir::dyn_cast<BlockArgument>(resIt.first))
                bufName = "buf" + std::to_string(barg.getArgNumber());
            std::vector<std::pair<int64_t, simt::semantics::SemValue>> entries;
            entries.reserve(resIt.second.size());
            for (const auto &kv : resIt.second)
                entries.push_back(kv);
            std::sort(entries.begin(), entries.end(),
                      [](const auto &a, const auto &b) { return a.first < b.first; });
            for (const auto &kv : entries)
                llvm::outs() << "  " << bufName << "[" << kv.first
                             << "] = " << kv.second.asInt64() << "\n";
        }
    }

    return 0;
}
