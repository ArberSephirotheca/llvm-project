// Minimal SIMT-Step -> HLSL raiser for a constrained subset.

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "simt-step/semantics/SimpleSemantics.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace {

struct HlslEmitter {
    std::string indent;
    llvm::DenseMap<Value, std::string> names;
    unsigned tmpId = 0;
    llvm::raw_ostream &os;

    explicit HlslEmitter(llvm::raw_ostream &os) : os(os) {}

    std::string makeTmp() { return "t" + std::to_string(tmpId++); }

    std::string get(Value v) const {
        if (auto it = names.find(v); it != names.end())
            return it->second;
        return "<unk>";
    }

    void emitIndent() { os << indent; }

    std::string emitValue(Value v) {
        if (auto it = names.find(v); it != names.end())
            return it->second;
        if (auto *def = v.getDefiningOp())
            return emitExpr(def);
        return "<unk>";
    }

    std::string emitExpr(Operation *op) {
        if (auto c = dyn_cast<arith::ConstantIntOp>(op)) {
            return std::to_string(c.value());
        }
        if (auto add = dyn_cast<arith::AddIOp>(op)) {
            return "(" + emitValue(add.getLhs()) + " + " + emitValue(add.getRhs()) + ")";
        }
        if (auto rem = dyn_cast<arith::RemSIOp>(op)) {
            return "(" + emitValue(rem.getLhs()) + " % " + emitValue(rem.getRhs()) + ")";
        }
        if (auto cmp = dyn_cast<arith::CmpIOp>(op)) {
            std::string pred;
            switch (cmp.getPredicate()) {
            case arith::CmpIPredicate::eq: pred = "=="; break;
            case arith::CmpIPredicate::ne: pred = "!="; break;
            case arith::CmpIPredicate::slt: pred = "<"; break;
            case arith::CmpIPredicate::sle: pred = "<="; break;
            case arith::CmpIPredicate::sgt: pred = ">"; break;
            case arith::CmpIPredicate::sge: pred = ">="; break;
            default: pred = "/*cmp*/"; break;
            }
            return "(" + emitValue(cmp.getLhs()) + " " + pred + " " + emitValue(cmp.getRhs()) + ")";
        }
        if (isa<simt::dialect::DispatchThreadIdOp>(op)) {
            return "tid.x";
        }
        if (auto wave = dyn_cast<simt::dialect::WaveCountBitsOp>(op)) {
            return "WaveActiveCountBits(" + emitValue(wave.getPredicate()) + ")";
        }
        return "<unsupported>";
    }

    LogicalResult emitIf(simt::dialect::IfOp ifOp) {
        if (ifOp.getNumResults() > 1)
            return failure();
        std::string resName;
        if (!ifOp.getResults().empty()) {
            resName = makeTmp();
            emitIndent();
            os << "int " << resName << ";\n";
        }
        emitIndent();
        os << "if (" << get(ifOp.getCondition()) << ") {\n";
        indent += "  ";
        if (failed(emitRegionAssign(ifOp.getThenRegion(), resName)))
            return failure();
        indent.pop_back(); indent.pop_back();
        emitIndent();
        os << "} else {\n";
        indent += "  ";
        if (failed(emitRegionAssign(ifOp.getElseRegion(), resName)))
            return failure();
        indent.pop_back(); indent.pop_back();
        emitIndent();
        os << "}\n";
        if (!resName.empty())
            names[ifOp.getResult(0)] = resName;
        return success();
    }

    LogicalResult emitRegionAssign(Region &r, const std::string &target) {
        auto &blk = r.front();
        for (auto &op : blk) {
            if (auto y = dyn_cast<simt::dialect::YieldOp>(op)) {
                if (!y.getOperands().empty() && !target.empty()) {
                    emitIndent();
                    os << target << " = " << get(y.getOperand(0)) << ";\n";
                }
                return success();
            }
            if (failed(emitOp(&op)))
                return failure();
        }
        return success();
    }

    LogicalResult emitLoop(simt::dialect::LoopOp loop) {
        // Expect two inits, prepare with cmp, body yield two carried values.
        if (loop.getInits().size() != 2 || loop.getResults().size() != 2)
            return failure();
        std::string accName = makeTmp();
        std::string iName = makeTmp();
        emitIndent();
        os << "int " << accName << " = " << emitValue(loop.getInits()[0]) << ";\n";
        emitIndent();
        os << "int " << iName << " = " << emitValue(loop.getInits()[1]) << ";\n";

        auto &prep = loop.getPrepareRegion().front();
        auto *condOp = prep.getTerminator();
        auto cond = dyn_cast<simt::dialect::ConditionOp>(condOp);
        if (!cond)
            return failure();
        names[prep.getArgument(0)] = accName;
        names[prep.getArgument(1)] = iName;

        emitIndent();
        os << "while (true) {\n";
        indent += "  ";
        std::string condExpr = emitValue(cond.getCondition());
        emitIndent();
        os << "if (!(" << condExpr << ")) break;\n";

        auto &body = loop.getBodyRegion().front();
        names[body.getArgument(0)] = accName;
        names[body.getArgument(1)] = iName;
        std::string nextAcc, nextI;
        for (auto &op : body) {
            if (auto y = dyn_cast<simt::dialect::YieldOp>(op)) {
                nextAcc = emitValue(y.getOperand(0));
                nextI = emitValue(y.getOperand(1));
                break;
            }
            if (failed(emitOp(&op)))
                return failure();
        }
        emitIndent();
        os << accName << " = " << nextAcc << ";\n";
        emitIndent();
        os << iName << " = " << nextI << ";\n";
        indent.pop_back(); indent.pop_back();
        emitIndent();
        os << "}\n";
        names[loop.getResult(0)] = accName;
        names[loop.getResult(1)] = iName;
        return success();
    }

    LogicalResult emitOp(Operation *op) {
        if (auto c = dyn_cast<arith::ConstantIntOp>(op)) {
            names[c.getResult()] = std::to_string(c.value());
            return success();
        }
        if (auto did = dyn_cast<simt::dialect::DispatchThreadIdOp>(op)) {
            std::string tmp = makeTmp();
            names[did.getResult()] = tmp;
            emitIndent();
            os << "int " << tmp << " = tid.x;\n";
            return success();
        }
        if (isa<arith::AddIOp, arith::RemSIOp, arith::CmpIOp,
                simt::dialect::WaveCountBitsOp>(op)) {
            std::string tmp = makeTmp();
            names[op->getResult(0)] = tmp;
            emitIndent();
            os << "int " << tmp << " = " << emitExpr(op) << ";\n";
            return success();
        }
        if (auto ifOp = dyn_cast<simt::dialect::IfOp>(op))
            return emitIf(ifOp);
        if (auto loop = dyn_cast<simt::dialect::LoopOp>(op))
            return emitLoop(loop);
        if (auto store = dyn_cast<simt::dialect::BufferStoreOp>(op)) {
            emitIndent();
            os << "buf" << store.getResource().cast<BlockArgument>().getArgNumber()
               << "[" << get(store.getIndex()) << "] = " << get(store.getValue()) << ";\n";
            return success();
        }
        if (isa<simt::dialect::YieldOp, simt::dialect::ConditionOp>(op))
            return success();
        return failure();
    }
};

} // namespace

int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);
    llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                         llvm::cl::desc("<input mlir>"),
                                         llvm::cl::init("-"));
    llvm::cl::ParseCommandLineOptions(argc, argv, "simt-step raise to HLSL\n");

    DialectRegistry registry;
    simt::dialect::registerSimtStepDialect(registry);
    registry.insert<arith::ArithDialect, func::FuncDialect>();
    MLIRContext context(registry);
    context.loadAllAvailableDialects();

    auto module = parseSourceFile<ModuleOp>(inputFile, &context);
    if (!module) {
        llvm::errs() << "failed to parse module\n";
        return 1;
    }
    func::FuncOp func;
    // Prefer @main via symbol lookup.
    if (auto sym = module->lookupSymbol<func::FuncOp>("main"))
        func = sym;
    // Else pick the first func.
    if (!func)
        for (auto op : module->getOps<func::FuncOp>()) {
            func = op;
            break;
        }
    if (!func) {
        llvm::errs() << "no func.func found\n";
        return 1;
    }

    // Emit HLSL.
    llvm::outs() << "[numthreads(";
    int64_t ntx = 1, nty = 1, ntz = 1;
    if (auto denseAttr = func->getAttrOfType<DenseI64ArrayAttr>("simt.num_threads")) {
        auto vals = denseAttr.asArrayRef();
        if (vals.size() == 3) {
            ntx = vals[0];
            nty = vals[1];
            ntz = vals[2];
        } else {
            llvm::errs() << "warning: simt.num_threads has unexpected size\n";
        }
    } else if (auto arr = func->getAttrOfType<ArrayAttr>("simt.num_threads")) {
        if (arr.size() == 3) {
            ntx = mlir::cast<IntegerAttr>(arr[0]).getInt();
            nty = mlir::cast<IntegerAttr>(arr[1]).getInt();
            ntz = mlir::cast<IntegerAttr>(arr[2]).getInt();
        } else {
            llvm::errs() << "warning: simt.num_threads has unexpected size\n";
        }
    } else {
        llvm::errs() << "warning: missing simt.num_threads on @main, defaulting to [1,1,1]\n";
    }
    llvm::outs() << ntx << "," << nty << "," << ntz << ")]\n";

    llvm::outs() << "void main(";
    SmallVector<BlockArgument> resources;
    for (auto arg : func.getArguments()) {
        auto ty = arg.getType();
        if (ty.isa<simt::dialect::ResourceType>())
            resources.push_back(arg);
    }
    for (unsigned i = 0; i < resources.size(); ++i) {
        llvm::outs() << "RWStructuredBuffer<int> buf" << i << " : register(u" << i << ")";
        if (i + 1 != resources.size())
            llvm::outs() << ", ";
    }
    if (!resources.empty())
        llvm::outs() << ", ";
    llvm::outs() << "uint3 tid : SV_DispatchThreadID) {\n";

    HlslEmitter emitter(llvm::outs());
    emitter.indent = "  ";
    auto &entry = func.getBody().front();
    for (auto &op : entry) {
        if (auto ret = dyn_cast<func::ReturnOp>(op))
            break;
        if (failed(emitter.emitOp(&op))) {
            llvm::errs() << "unsupported op in raiser: " << op.getName() << "\n";
            return 1;
        }
    }
    llvm::outs() << "}\n";
    return 0;
}
