// Shared HLSL emitter for SIMT-Step MLIR kernels.
// Derived from the simt-step-raise tool; supports the subset used by the fuzzer.

#include "HlslEmitter.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Block.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace simt::raise {
namespace {

static std::string formatConst(arith::ConstantIntOp c) {
    int64_t v = c.value();
    if (auto intTy = mlir::dyn_cast<IntegerType>(c.getType())) {
        if (intTy.getWidth() == 1)
            return v ? "true" : "false";
    }
    return std::to_string(v);
}

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
            return formatConst(c);
        }
        if (auto add = dyn_cast<arith::AddIOp>(op)) {
            return "(" + emitValue(add.getLhs()) + " + " + emitValue(add.getRhs()) + ")";
        }
        if (auto sub = dyn_cast<arith::SubIOp>(op)) {
            return "(" + emitValue(sub.getLhs()) + " - " + emitValue(sub.getRhs()) + ")";
        }
        if (auto mul = dyn_cast<arith::MulIOp>(op)) {
            return "(" + emitValue(mul.getLhs()) + " * " + emitValue(mul.getRhs()) + ")";
        }
        if (auto andi = dyn_cast<arith::AndIOp>(op)) {
            return "(" + emitValue(andi.getLhs()) + " & " + emitValue(andi.getRhs()) + ")";
        }
        if (auto ori = dyn_cast<arith::OrIOp>(op)) {
            return "(" + emitValue(ori.getLhs()) + " | " + emitValue(ori.getRhs()) + ")";
        }
        if (auto shl = dyn_cast<arith::ShLIOp>(op)) {
            return "(" + emitValue(shl.getLhs()) + " << " + emitValue(shl.getRhs()) + ")";
        }
        if (auto shr = dyn_cast<arith::ShRSIOp>(op)) {
            return "(" + emitValue(shr.getLhs()) + " >> " + emitValue(shr.getRhs()) + ")";
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

    LogicalResult emitRegionAssign(Region &r,
                                   llvm::ArrayRef<std::string> targets) {
        auto &blk = r.front();
        for (auto &op : blk) {
            if (auto y = dyn_cast<simt::dialect::YieldOp>(op)) {
                if (targets.empty()) {
                    if (!y.getOperands().empty())
                        return failure();
                } else if (y.getNumOperands() != targets.size()) {
                    return failure();
                }
                for (unsigned i = 0; i < targets.size(); ++i) {
                    if (targets[i].empty())
                        continue;
                    emitIndent();
                    os << targets[i] << " = " << get(y.getOperand(i)) << ";\n";
                }
                return success();
            }
            if (auto br = dyn_cast<simt::dialect::BreakOp>(op)) {
                (void)br;
                emitIndent();
                os << "break;\n";
                return success();
            }
            if (failed(emitOp(&op)))
                return failure();
        }
        return success();
    }

    LogicalResult emitIf(simt::dialect::IfOp ifOp) {
        unsigned numResults = ifOp.getNumResults();
        std::vector<std::string> resultNames;
        resultNames.reserve(numResults);
        for (unsigned i = 0; i < numResults; ++i) {
            std::string name = makeTmp();
            resultNames.push_back(name);
            emitIndent();
            os << emitType(ifOp.getResultTypes()[i]) << " " << name << ";\n";
        }
        emitIndent();
        os << "if (" << get(ifOp.getCondition()) << ") {\n";
        indent += "  ";
        if (failed(emitRegionAssign(ifOp.getThenRegion(), resultNames)))
            return failure();
        indent.pop_back();
        indent.pop_back();
        bool hasElseRegion = !ifOp.getElseRegion().empty();
        if (numResults > 0 && (!hasElseRegion || ifOp.getElseRegion().front().empty()))
            return failure();
        bool hasElse = hasElseRegion && !ifOp.getElseRegion().front().empty();
        if (hasElse) {
            emitIndent();
            os << "} else {\n";
            indent += "  ";
            if (failed(emitRegionAssign(ifOp.getElseRegion(), resultNames)))
                return failure();
            indent.pop_back();
            indent.pop_back();
            emitIndent();
            os << "}\n";
        } else {
            emitIndent();
            os << "}\n";
        }
        for (unsigned i = 0; i < numResults; ++i)
            names[ifOp.getResult(i)] = resultNames[i];
        return success();
    }

    LogicalResult emitLoop(simt::dialect::LoopOp loop) {
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

        // Declare induction var and use a while loop with explicit break.
        emitIndent();
        os << emitType(loop.getInits()[1].getType()) << " " << iName << " = " << initI
           << ";\n";
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
        enum class TermKind { Yield, Continue, Break };
        TermKind term = TermKind::Yield;
        for (auto &op : body) {
            if (auto y = dyn_cast<simt::dialect::YieldOp>(op)) {
                nextAcc = emitValue(y.getOperand(0));
                nextI = emitValue(y.getOperand(1));
                term = TermKind::Yield;
                break;
            }
            if (auto cont = dyn_cast<simt::dialect::ContinueOp>(op)) {
                nextAcc = emitValue(cont.getOperand(0));
                nextI = emitValue(cont.getOperand(1));
                emitIndent();
                os << accName << " = " << nextAcc << ";\n";
                emitIndent();
                os << iName << " = " << nextI << ";\n";
                emitIndent();
                os << "continue;\n";
                term = TermKind::Continue;
                break;
            }
            if (auto br = dyn_cast<simt::dialect::BreakOp>(op)) {
                nextAcc = emitValue(br.getOperand(0));
                nextI = emitValue(br.getOperand(1));
                emitIndent();
                os << accName << " = " << nextAcc << ";\n";
                emitIndent();
                os << iName << " = " << nextI << ";\n";
                emitIndent();
                os << "break;\n";
                term = TermKind::Break;
                break;
            }
            if (failed(emitOp(&op)))
                return failure();
        }
        if (term == TermKind::Yield) {
            emitIndent();
            os << accName << " = " << nextAcc << ";\n";
            emitIndent();
            os << iName << " = " << nextI << ";\n";
        }
        indent.pop_back();
        indent.pop_back();
        emitIndent();
        os << "}\n";
        names[loop.getResult(0)] = accName;
        names[loop.getResult(1)] = iName;
        return success();
    }

    LogicalResult emitSwitch(simt::dialect::SwitchOp sw) {
        unsigned numResults = sw.getNumResults();
        auto initVals = sw.getInitialValues();
        std::vector<std::string> resultNames;
        resultNames.reserve(numResults);

        for (unsigned i = 0; i < numResults; ++i) {
            std::string name = makeTmp();
            resultNames.push_back(name);
            emitIndent();
            os << emitType(sw.getResultTypes()[i]) << " " << name;
            if (i < initVals.size())
                os << " = " << get(initVals[i]);
            os << ";\n";
        }

        bool hasFallthroughAttr = false;
        std::vector<bool> caseFallthrough;
        caseFallthrough.reserve(sw.getCaseBody().getBlocks().size());
        for (auto &blk : sw.getCaseBody()) {
            bool fall = false;
            if (!blk.empty()) {
                if (auto y = dyn_cast<simt::dialect::YieldOp>(blk.back())) {
                    if (auto attr = y->getAttrOfType<mlir::BoolAttr>("fallthrough")) {
                        hasFallthroughAttr = true;
                        fall = attr.getValue();
                    }
                }
            }
            caseFallthrough.push_back(fall);
        }

        bool isLegacyFallthroughSwitch = false;
        if (!hasFallthroughAttr && numResults >= 4) {
            auto types = sw.getResultTypes();
            auto isI1 = [](mlir::Type ty) {
                if (auto intTy = mlir::dyn_cast<IntegerType>(ty))
                    return intTy.getWidth() == 1;
                return false;
            };
            isLegacyFallthroughSwitch = isI1(types[numResults - 1]) &&
                                        isI1(types[numResults - 2]) &&
                                        isI1(types[numResults - 3]);
        }

        if (isLegacyFallthroughSwitch) {
            auto &region = sw.getCaseBody();
            for (auto &blk : region) {
                emitIndent();
                os << "{\n";
                indent += "  ";
                for (unsigned i = 0; i < blk.getNumArguments(); ++i) {
                    if (i < resultNames.size())
                        names[blk.getArgument(i)] = resultNames[i];
                }
                bool sawYield = false;
                for (auto &op : blk) {
                    if (auto y = dyn_cast<simt::dialect::YieldOp>(op)) {
                        if (y.getNumOperands() != numResults)
                            return failure();
                        for (unsigned i = 0; i < numResults; ++i) {
                            emitIndent();
                            os << resultNames[i] << " = " << get(y.getOperand(i)) << ";\n";
                        }
                        sawYield = true;
                        break;
                    }
                    if (failed(emitOp(&op)))
                        return failure();
                }
                if (!sawYield)
                    return failure();
                indent.pop_back();
                indent.pop_back();
                emitIndent();
                os << "}\n";
            }
            for (unsigned i = 0; i < numResults; ++i)
                names[sw.getResult(i)] = resultNames[i];
            return success();
        }

        emitIndent();
        os << "switch (" << get(sw.getSelector()) << ") {\n";
        indent += "  ";

        auto caseValues = sw.getCaseValues();
        auto &region = sw.getCaseBody();
        unsigned numBlocks =
            static_cast<unsigned>(std::distance(region.begin(), region.end()));
        unsigned blockIndex = 0;

        for (auto &blk : region) {
            bool isDefault = (blockIndex + 1 == numBlocks);
            emitIndent();
            if (isDefault)
                os << "default:\n";
            else
                os << "case " << caseValues[blockIndex] << ":\n";
            indent += "  ";
            emitIndent();
            os << "{\n";
            indent += "  ";

            for (unsigned i = 0; i < blk.getNumArguments(); ++i) {
                if (i < resultNames.size())
                    names[blk.getArgument(i)] = resultNames[i];
            }

            bool sawYield = false;
            for (auto &op : blk) {
                if (auto y = dyn_cast<simt::dialect::YieldOp>(op)) {
                    if (y.getNumOperands() != numResults)
                        return failure();
                    for (unsigned i = 0; i < numResults; ++i) {
                        emitIndent();
                        os << resultNames[i] << " = " << get(y.getOperand(i)) << ";\n";
                    }
                    sawYield = true;
                    break;
                }
                if (failed(emitOp(&op)))
                    return failure();
            }
            if (!sawYield)
                return failure();
            bool fallthrough = false;
            if (blockIndex < caseFallthrough.size())
                fallthrough = caseFallthrough[blockIndex];
            if (!fallthrough || isDefault) {
                emitIndent();
                os << "break;\n";
            }

            indent.pop_back();
            indent.pop_back();
            emitIndent();
            os << "}\n";
            indent.pop_back();
            indent.pop_back();
            ++blockIndex;
        }

        indent.pop_back();
        indent.pop_back();
        emitIndent();
        os << "}\n";

        for (unsigned i = 0; i < numResults; ++i)
            names[sw.getResult(i)] = resultNames[i];
        return success();
    }

    LogicalResult emitOp(Operation *op) {
        if (auto c = dyn_cast<arith::ConstantIntOp>(op)) {
            names[c.getResult()] = formatConst(c);
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
                arith::SubIOp, arith::MulIOp, arith::AndIOp, arith::OrIOp,
                arith::ShLIOp, arith::ShRSIOp, simt::dialect::WaveCountBitsOp>(op)) {
            std::string tmp = makeTmp();
            names[op->getResult(0)] = tmp;
            emitIndent();
            os << emitType(op->getResult(0).getType()) << " " << tmp << " = " << emitExpr(op) << ";\n";
            return success();
        }
        if (auto br = dyn_cast<simt::dialect::BreakOp>(op)) {
            (void)br;
            emitIndent();
            os << "break;\n";
            return success();
        }
        if (auto ifOp = dyn_cast<simt::dialect::IfOp>(op))
            return emitIf(ifOp);
        if (auto loop = dyn_cast<simt::dialect::LoopOp>(op))
            return emitLoop(loop);
        if (auto sw = dyn_cast<simt::dialect::SwitchOp>(op))
            return emitSwitch(sw);
        if (auto load = dyn_cast<simt::dialect::BufferLoadOp>(op)) {
            std::string tmp = makeTmp();
            names[load.getResult()] = tmp;
            emitIndent();
            os << emitType(load.getResult().getType()) << " " << tmp << " = "
               << "buf"
               << mlir::cast<BlockArgument>(load.getResource()).getArgNumber()
               << "[" << get(load.getIndex()) << "];\n";
            return success();
        }
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

LogicalResult emitModuleAsHlsl(ModuleOp module, llvm::raw_ostream &os) {
    func::FuncOp func;
    if (auto sym = module.lookupSymbol<func::FuncOp>("main"))
        func = sym;
    if (!func)
        for (auto op : module.getOps<func::FuncOp>()) {
            func = op;
            break;
        }
    if (!func) {
        os << "no func.func found\n";
        return failure();
    }

    int64_t ntx = 1, nty = 1, ntz = 1;
    if (auto attr = func->getAttr("simt.num_threads")) {
        if (auto denseAttr = mlir::dyn_cast<DenseI64ArrayAttr>(attr)) {
            auto vals = denseAttr.asArrayRef();
            if (vals.size() == 3) {
                ntx = vals[0];
                nty = vals[1];
                ntz = vals[2];
            }
        } else if (auto arrayAttr = mlir::dyn_cast<ArrayAttr>(attr)) {
            if (arrayAttr.size() == 3) {
                auto x = mlir::dyn_cast<IntegerAttr>(arrayAttr[0]);
                auto y = mlir::dyn_cast<IntegerAttr>(arrayAttr[1]);
                auto z = mlir::dyn_cast<IntegerAttr>(arrayAttr[2]);
                if (x && y && z) {
                    ntx = x.getInt();
                    nty = y.getInt();
                    ntz = z.getInt();
                }
            }
        }
    }

    SmallVector<BlockArgument> resources;
    for (auto arg : func.getArguments()) {
        if (mlir::isa<simt::dialect::ResourceType>(arg.getType()))
            resources.push_back(arg);
    }
    for (unsigned i = 0; i < resources.size(); ++i) {
        os << "RWStructuredBuffer<" << "int" << "> buf" << i
           << " : register(u" << i << ");\n";
    }
    if (!resources.empty())
        os << "\n";

    os << "[numthreads(" << ntx << "," << nty << "," << ntz << ")]\n";
    os << "void main(";
    bool first = true;
    for (auto arg : func.getArguments()) {
        if (mlir::isa<simt::dialect::ResourceType>(arg.getType()))
            continue;
        if (!first)
            os << ", ";
        first = false;
        if (auto intTy = mlir::dyn_cast<IntegerType>(arg.getType())) {
            unsigned w = intTy.getWidth();
            if (w == 1)
                os << "bool";
            else if (w == 32)
                os << "uint";
            else
                os << "int";
        } else if (arg.getType().isIndex()) {
            os << "uint";
        } else {
            os << "int";
        }
        os << " arg" << arg.getArgNumber();
    }
    if (!first)
        os << ", ";
    os << "uint3 tid : SV_DispatchThreadID) {\n";

    HlslEmitter emitter(os);
    emitter.indent = "  ";
    for (unsigned i = 0; i < resources.size(); ++i)
        emitter.names[resources[i]] = "buf" + std::to_string(i);
    for (auto arg : func.getArguments()) {
        if (mlir::isa<simt::dialect::ResourceType>(arg.getType()))
            continue;
        emitter.names[arg] = "arg" + std::to_string(arg.getArgNumber());
    }
    auto &entry = func.getBody().front();
    for (auto &op : entry) {
        if (auto ret = dyn_cast<func::ReturnOp>(op))
            break;
        if (failed(emitter.emitOp(&op))) {
            os << "unsupported op in raiser: " << op.getName() << "\n";
            return failure();
        }
    }
    os << "}\n";
    return success();
}

} // namespace simt::raise
