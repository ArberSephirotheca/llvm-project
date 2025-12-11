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

#include <optional>

using namespace mlir;

namespace {

struct HlslEmitter {
    std::string indent;
    llvm::DenseMap<Value, std::string> names;
    unsigned tmpId = 0;
    llvm::raw_ostream &os;

    explicit HlslEmitter(llvm::raw_ostream &os) : os(os) {}

    std::string makeTmp() { return "t" + std::to_string(tmpId++); }

    std::string emitType(Type ty) const {
        if (auto it = mlir::dyn_cast<IntegerType>(ty)) {
            unsigned w = it.getWidth();
            if (w == 1)
                return "bool";
            if (w == 32)
                return "uint";
            return "int";
        }
        if (ty.isIndex())
            return "uint";
        return "int";
    }

    std::string typeFor(Value v) const { return emitType(v.getType()); }

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
        auto trivialElseYield = [&]() -> std::optional<std::string> {
            auto &blk = ifOp.getElseRegion().front();
            if (blk.empty())
                return std::string{};
            if (std::next(blk.begin()) != blk.end())
                return std::nullopt;
            if (auto y = dyn_cast<simt::dialect::YieldOp>(&blk.front())) {
                if (y.getNumOperands() == 0)
                    return std::string{};
                if (y.getNumOperands() == 1)
                    return emitValue(y.getOperand(0));
            }
            return std::nullopt;
        };
        std::string resName;
        std::optional<std::string> elseInit;
        bool hasResult = !ifOp.getResults().empty();
        if (hasResult) {
            resName = makeTmp();
            elseInit = trivialElseYield();
            emitIndent();
            os << emitType(ifOp.getResultTypes().front()) << " " << resName;
            if (elseInit)
                os << " = " << *elseInit;
            os << ";\n";
        }
        emitIndent();
        os << "if (" << get(ifOp.getCondition()) << ") {\n";
        indent += "  ";
        if (failed(emitRegionAssign(ifOp.getThenRegion(), resName)))
            return failure();
        indent.pop_back(); indent.pop_back();
        bool omitElse = (!hasResult && trivialElseYield().has_value()) ||
                        (hasResult && elseInit.has_value());
        if (!omitElse) {
            emitIndent();
            os << "} else {\n";
            indent += "  ";
            if (failed(emitRegionAssign(ifOp.getElseRegion(), resName)))
                return failure();
            indent.pop_back(); indent.pop_back();
            emitIndent();
            os << "}\n";
        } else {
            emitIndent();
            os << "}\n";
        }
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
        os << emitType(loop.getInits()[0].getType()) << " " << accName
           << " = " << emitValue(loop.getInits()[0]) << ";\n";
        std::string initI = emitValue(loop.getInits()[1]);

        auto &prep = loop.getPrepareRegion().front();
        auto *condOp = prep.getTerminator();
        auto cond = dyn_cast<simt::dialect::ConditionOp>(condOp);
        if (!cond)
            return failure();
        names[prep.getArgument(0)] = accName;
        names[prep.getArgument(1)] = iName;

        // Try to recognize a simple for-loop shape: cmp on the induction variable
        // and an add with a constant step.
        bool canUseFor = false;
        std::string condLHS, condRHS, stepExpr;
        arith::CmpIPredicate cmpPred = arith::CmpIPredicate::eq;
        if (auto cmp = dyn_cast_or_null<arith::CmpIOp>(
                cond.getCondition().getDefiningOp())) {
            cmpPred = cmp.getPredicate();
            auto lhs = cmp.getLhs();
            auto rhs = cmp.getRhs();
            auto prepIdx = prep.getArgument(1);
            auto prepIdxStr = iName;
            auto otherStr = [&](Value v) { return emitValue(v); };
            if (lhs == prepIdx || rhs == prepIdx) {
                condLHS = (lhs == prepIdx) ? prepIdxStr : otherStr(lhs);
                condRHS = (rhs == prepIdx) ? prepIdxStr : otherStr(rhs);
                // Check the body step: yield idx' = idx +/- C
                auto &body = loop.getBodyRegion().front();
                names[body.getArgument(0)] = accName;
                names[body.getArgument(1)] = iName;
                for (auto &op : body) {
                    if (auto y = dyn_cast<simt::dialect::YieldOp>(op)) {
                        auto nextIdx = y.getOperand(1);
                        if (auto add = dyn_cast_or_null<arith::AddIOp>(
                                nextIdx.getDefiningOp())) {
                            if (add.getLhs() == body.getArgument(1)) {
                                stepExpr = emitValue(add.getRhs());
                                canUseFor = true;
                            } else if (add.getRhs() == body.getArgument(1)) {
                                stepExpr = emitValue(add.getLhs());
                                canUseFor = true;
                            }
                        }
                        break;
                    }
                }
            }
        }

        emitIndent();
        if (canUseFor) {
            // Emit for-loop header.
            os << "for (" << emitType(loop.getInits()[1].getType()) << " " << iName
               << " = " << initI
               << "; (" << condLHS;
            switch (cmpPred) {
            case arith::CmpIPredicate::slt: os << " < "; break;
            case arith::CmpIPredicate::sle: os << " <= "; break;
            case arith::CmpIPredicate::sgt: os << " > "; break;
            case arith::CmpIPredicate::sge: os << " >= "; break;
            case arith::CmpIPredicate::ne: os << " != "; break;
            case arith::CmpIPredicate::eq: os << " == "; break;
            default: os << " /*cmp*/ "; break;
            }
            os << condRHS << "); " << iName << " = " << iName << " + " << stepExpr
               << ") {\n";
        } else {
            os << emitType(loop.getInits()[1].getType()) << " " << iName << " = " << initI << ";\n";
            emitIndent();
            os << "while (true) {\n";
        }
        indent += "  ";
        if (!canUseFor) {
            std::string condExpr = emitValue(cond.getCondition());
            emitIndent();
            os << "if (!(" << condExpr << ")) break;\n";
        }

        // Body emission (re-run with established names to avoid stale mapping).
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
        if (!canUseFor) {
            emitIndent();
            os << iName << " = " << nextI << ";\n";
        }
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
            os << emitType(did.getResult().getType()) << " " << tmp << " = tid.x;\n";
            return success();
        }
        if (isa<arith::AddIOp, arith::RemSIOp, arith::CmpIOp,
                simt::dialect::WaveCountBitsOp>(op)) {
            std::string tmp = makeTmp();
            names[op->getResult(0)] = tmp;
            emitIndent();
            os << emitType(op->getResult(0).getType()) << " " << tmp << " = " << emitExpr(op) << ";\n";
            return success();
        }
        if (auto ifOp = dyn_cast<simt::dialect::IfOp>(op))
            return emitIf(ifOp);
        if (auto loop = dyn_cast<simt::dialect::LoopOp>(op))
            return emitLoop(loop);
        if (auto store = dyn_cast<simt::dialect::BufferStoreOp>(op)) {
            emitIndent();
            os << "buf" << mlir::cast<BlockArgument>(store.getResource()).getArgNumber()
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
        if (mlir::isa<simt::dialect::ResourceType>(ty))
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
