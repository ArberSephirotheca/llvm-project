#include "SimtProgramGenerator.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Parser/Parser.h>

#include <cstdint>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/raw_ostream.h>

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

using namespace mlir;

namespace simt::fuzz {

static std::string buildKernelText(const GeneratorConfig &cfg) {
    llvm::SmallString<256> storage;
    llvm::raw_svector_ostream os(storage);
    os << "module {\n";
    os << "  func.func @main()";
    os << " attributes {simt.num_threads = array<i64: " << cfg.numThreads[0]
       << ", " << cfg.numThreads[1] << ", " << cfg.numThreads[2] << ">} {\n";
    os << "    %tid = \"simt_step.dispatch_thread_id\"() : () -> i32\n";
    os << "    %c0 = arith.constant 0 : i32\n";
    os << "    %cond = arith.cmpi eq, %tid, %c0 : i32\n";
    os << "    %val = \"simt_step.if\"(%cond) ({\n";
    os << "      %zero = arith.constant 0 : i32\n";
    os << "      %c0_i32 = arith.constant 0 : i32\n";
    os << "      %loop_res:2 = \"simt_step.loop\"(%zero, %c0_i32) ({\n";
    os << "      ^bb0(%acc: i32, %i: i32):\n";
    os << "        %c4_i32 = arith.constant 4 : i32\n";
    os << "        %lt = arith.cmpi slt, %i, %c4_i32 : i32\n";
    os << "        \"simt_step.condition\"(%lt, %acc, %i) : (i1, i32, i32) -> ()\n";
    os << "      }, {\n";
    os << "      ^bb0(%acc: i32, %i: i32):\n";
    os << "        %sum = arith.addi %acc, %i : i32\n";
    os << "        %c1_i32 = arith.constant 1 : i32\n";
    os << "        %next = arith.addi %i, %c1_i32 : i32\n";
    os << "        \"simt_step.yield\"(%sum, %next) : (i32, i32) -> ()\n";
    os << "      }) : (i32, i32) -> (i32, i32)\n";
    os << "      \"simt_step.yield\"(%loop_res#0) : (i32) -> ()\n";
    os << "    }, {\n";
    os << "      \"simt_step.yield\"(%tid) : (i32) -> ()\n";
    os << "    }) : (i1) -> i32\n";
    os << "    func.return\n";
    os << "  }\n";
    os << "}\n";
    return std::string(os.str());
}

mlir::OwningOpRef<mlir::ModuleOp>
createDeterministicIfLoopModule(mlir::MLIRContext &context,
                                const GeneratorConfig &cfg) {
    DialectRegistry registry;
    simt::dialect::registerSimtStepDialect(registry);
    registry.insert<arith::ArithDialect, func::FuncDialect>();
    context.appendDialectRegistry(registry);
    (void)context.getOrLoadDialect<simt::dialect::SimtStepDialect>();
    (void)context.getOrLoadDialect<arith::ArithDialect>();
    (void)context.getOrLoadDialect<func::FuncDialect>();

    std::string text = buildKernelText(cfg);
    return parseSourceString<ModuleOp>(text, &context);
}

} // namespace simt::fuzz
