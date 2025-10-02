#include "simt-step/semantics/StructuredProgram.h"

#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassRegistry.h>

namespace simt::semantics {

namespace {

struct DumpStructuredProgramPass
    : public mlir::PassWrapper<DumpStructuredProgramPass, mlir::OperationPass<mlir::ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DumpStructuredProgramPass)

    llvm::StringRef getArgument() const final { return "simt-dump-structured-program"; }
    llvm::StringRef getDescription() const final {
        return "Prints summary information about simt_struct blocks";
    }

    void runOnOperation() override {
        mlir::ModuleOp module = getOperation();
        StructuredProgram program;
        program.initialize(module);

        auto &os = llvm::outs();
        os << "entry: ";
        if (program.hasEntrySymbol())
            os << program.getEntrySymbol();
        else
            os << "<none>";
        os << '\n';

        for (const BlockInfo &info : program.blocks()) {
            os << "block " << info.symbol;
            os << " args=" << info.getArgumentCount();
            if (info.hasMergeTarget())
                os << " merge=" << info.mergeTarget.getValue();
            if (info.hasContinueTarget())
                os << " continue=" << info.continueTarget.getValue();
            os << '\n';
        }
    }
};

} // namespace

void registerDumpStructuredProgramPass() {
    static mlir::PassRegistration<DumpStructuredProgramPass> pass;
}

} // namespace simt::semantics
