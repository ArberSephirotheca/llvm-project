#ifndef SIMT_STEP_CONVERSION_STRUCTURED_CFGBUILDER_H
#define SIMT_STEP_CONVERSION_STRUCTURED_CFGBUILDER_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "simt-step/Dialect/SimtStructured/StructuredOps.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <string>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

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
  struct IfInfo;
  struct LoopInfo;
  struct SwitchInfo;

  struct EdgeInfo {
    BlockInfo *source = nullptr;
    BlockInfo *dest = nullptr;
    llvm::SmallVector<mlir::Value, 8> payload;
    llvm::SmallVector<mlir::Value, 4> maskValues;
    enum Kind { Plain, ConditionalTrue, ConditionalFalse, LoopBackEdge } kind =
        Plain;
    mlir::Value condition;
    mlir::Value switchDoneFlag;
    bool isSwitchFallthrough = false;
    mlir::Operation *origin = nullptr;
  };

  struct IfInfo {
    mlir::Operation *op = nullptr;
    BlockInfo *parent = nullptr;
    BlockInfo *thenBlock = nullptr;
    BlockInfo *elseBlock = nullptr;
    mlir::Value condition;
  };

  struct LoopInfo {
    mlir::Operation *op = nullptr;
    BlockInfo *parent = nullptr;
    BlockInfo *prepareBlock = nullptr;
    BlockInfo *bodyBlock = nullptr;
    mlir::Value condition;
    llvm::SmallVector<mlir::Value, 4> forwardedToBody;
    llvm::SmallVector<mlir::Value, 4> forwardedToExit;
  };

  struct SwitchInfo {
    mlir::Operation *op = nullptr;
    BlockInfo *parent = nullptr;
    llvm::SmallVector<BlockInfo *, 4> caseBlocks;
    BlockInfo *defaultBlock = nullptr;
    llvm::SmallVector<int64_t, 8> caseValues;
    unsigned payloadCount = 0;
    bool hasControlFlags = false;
    struct CaseRecord {
      BlockInfo *block = nullptr;
      BlockInfo *nextCase = nullptr;
      mlir::Value matchSeen;
      mlir::Value fallthrough;
      mlir::Value switchDone;
    };
    llvm::SmallVector<CaseRecord, 4> caseRecords;
    CaseRecord defaultRecord;
  };

  struct BlockInfo {
    mlir::Block *original = nullptr;
    llvm::SmallVector<mlir::Type, 4> carriedTypes;
    llvm::SmallVector<mlir::Value, 4> payloadSeed;
    llvm::SmallVector<mlir::BlockArgument, 4> blockArgs;
    llvm::SmallVector<mlir::Operation *, 4> controlOps;

    mlir::Operation *originalTerminator = nullptr;

    simt::structured::BlockOp structuredOp;
    mlir::Block *structuredBody = nullptr;
    mlir::BlockArgument structuredMaskArg;
    mlir::Value currentMask;
    llvm::SmallVector<mlir::BlockArgument, 4> structuredArgs;

    mlir::Block *mergeTarget = nullptr;
    mlir::Block *continueTarget = nullptr;

    bool requestsMaskPush = false;
    bool requestsMaskPop = false;
    std::string symbolName;
    llvm::SmallVector<const EdgeInfo *, 4> outgoingEdges;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<const EdgeInfo *, 2>>
        perOpEdges;
  };

  mlir::LogicalResult analyseBlocks();
  mlir::LogicalResult computePayloads();
  mlir::LogicalResult enumerateEdges();
  mlir::LogicalResult emitStructuredBlocks();
  mlir::LogicalResult cleanupOriginalCFG();

  mlir::LogicalResult emitStructuredBlock(BlockInfo &info);
  mlir::LogicalResult emitStructuredTerminator(BlockInfo &source);

  /// Helpers used while analysing structured control ops.
  mlir::LogicalResult analyseIfOp(BlockInfo &header, mlir::Operation *op);
  mlir::LogicalResult analyseLoopOp(BlockInfo &header, mlir::Operation *op);
  mlir::LogicalResult analyseSwitchOp(BlockInfo &header, mlir::Operation *op);

  /// Pull original blocks in source order so we can map them back later.
  void collectOriginalBlocks();

  BlockInfo &getOrCreateBlockInfo(mlir::Block *block);
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

  llvm::DenseMap<mlir::Operation *, IfInfo> ifInfos;
  llvm::DenseMap<mlir::Operation *, LoopInfo> loopInfos;
  llvm::DenseMap<mlir::Operation *, SwitchInfo> switchInfos;

  llvm::SmallVector<mlir::Value, 4> functionReturnValues;
  bool hasFunctionReturn = false;
};

} // namespace conversion
} // namespace simt

#endif // SIMT_STEP_CONVERSION_STRUCTURED_CFGBUILDER_H
