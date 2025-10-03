#ifndef SIMT_STEP_CONVERSION_STRUCTURED_CFGBUILDER_H
#define SIMT_STEP_CONVERSION_STRUCTURED_CFGBUILDER_H

#include "mlir/Support/LogicalResult.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

namespace mlir {
class Block;
class DominanceInfo;
class IRMapping;
class Operation;
class Region;
class Type;
class Value;
class ValueRange;
class FunctionOpInterface;
namespace func {
class FuncOp;
}
} // namespace mlir

namespace simt {
namespace conversion {

/// Forward declaration of the structured CFG builder.
///
/// The builder will eventually replace the ad-hoc lowering helpers in
/// `SimtStepToStructured.cpp`.  It takes a `func.func` that contains the
/// high-level `simt_step` control flow and materialises the structured SIMT
/// form in one pass.
class StructuredCFGBuilder {
public:
  explicit StructuredCFGBuilder(mlir::FunctionOpInterface func);

  /// Execute the structured lowering.  The initial implementation is a stub so
  /// the boilerplate can land independently from the functional rewrite.
  mlir::LogicalResult build();

private:
  struct BlockInfo;
  struct EdgeInfo;
  struct SwitchCaseInfo;

  mlir::LogicalResult analyseBlocks();
  mlir::LogicalResult computePayloads();
  mlir::LogicalResult enumerateEdges();
  mlir::LogicalResult emitStructuredBlocks();
  mlir::LogicalResult cleanupOriginalCFG();

  mlir::LogicalResult emitStructuredBlock(BlockInfo &info);
  mlir::LogicalResult emitStructuredTerminator(BlockInfo &source,
                                                const EdgeInfo &edge);

  /// Helpers used while analysing structured control ops.
  mlir::LogicalResult analyseIfOp(BlockInfo &header, mlir::Operation *op);
  mlir::LogicalResult analyseLoopOp(BlockInfo &header, mlir::Operation *op);
  mlir::LogicalResult analyseSwitchOp(BlockInfo &header, mlir::Operation *op);

  /// Pull original blocks in source order so we can map them back later.
  void collectOriginalBlocks();

  BlockInfo *lookupBlockInfo(mlir::Block *block);
  const BlockInfo *lookupBlockInfo(mlir::Block *block) const;

  /// Structured payload helpers.
  mlir::LogicalResult ensurePayloadShape(EdgeInfo &edge);
  mlir::LogicalResult propagatePayload(BlockInfo &source,
                                       BlockInfo &dest,
                                       llvm::ArrayRef<mlir::Value> values);

  /// Mask helpers used while emitting mask push/pop ops.
  mlir::LogicalResult materialiseMaskEntry(BlockInfo &info);
  mlir::LogicalResult materialiseMaskExit(BlockInfo &info);

  mlir::FunctionOpInterface func;
  llvm::SmallVector<mlir::Block *> blockOrder;

  /// Mapping from original blocks to collected metadata.
  llvm::DenseMap<mlir::Block *, BlockInfo> blockInfos;
  llvm::SmallVector<EdgeInfo> edges;

  /// Scratch storage used while cloning ops into structured blocks.
  std::unique_ptr<mlir::IRMapping> mapper;
  std::unique_ptr<mlir::DominanceInfo> domInfo;
};

} // namespace conversion
} // namespace simt

#endif // SIMT_STEP_CONVERSION_STRUCTURED_CFGBUILDER_H
