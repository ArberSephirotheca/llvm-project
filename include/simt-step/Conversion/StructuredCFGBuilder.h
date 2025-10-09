#ifndef SIMT_STEP_CONVERSION_STRUCTURED_CFGBUILDER_H
#define SIMT_STEP_CONVERSION_STRUCTURED_CFGBUILDER_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <string>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

namespace simt::structured {
class BlockOp;
class BranchOp;
class CondBranchOp;
class ReturnOp;
} // namespace simt::structured

namespace simt::conversion {

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
  /// Lightweight expression tree modelling lane masks.  Mask expressions are
  /// built during analysis and propagated alongside the structured CFG so we
  /// can reason about dynamic blocks without relying on push/pop scaffolding.
  struct MaskExpr {
    enum class Kind : uint8_t { Invalid, Full, Empty, Value, And, Or, Not };

    struct Node {
      Kind kind = Kind::Invalid;
      mlir::Value value;
      std::shared_ptr<const Node> lhs;
      std::shared_ptr<const Node> rhs;
    };

    MaskExpr() = default;
    explicit MaskExpr(std::shared_ptr<const Node> node) : node(std::move(node)) {}

    static MaskExpr full();
    static MaskExpr empty();
    static MaskExpr value(mlir::Value v);
    static MaskExpr makeAnd(const MaskExpr &lhs, const MaskExpr &rhs);
    static MaskExpr makeOr(const MaskExpr &lhs, const MaskExpr &rhs);
    static MaskExpr makeNot(const MaskExpr &arg);

    /// Returns a simplified version of the expression (constant folds basic
    /// cases).
  MaskExpr simplify() const;

    bool isValid() const { return static_cast<bool>(node); }
    Kind getKind() const { return node ? node->kind : Kind::Invalid; }

    mlir::Value getValue() const { return node ? node->value : mlir::Value(); }
    const MaskExpr getLHS() const;
    const MaskExpr getRHS() const;

    bool equals(const MaskExpr &other) const;
    bool operator==(const MaskExpr &other) const { return equals(other); }
    bool operator!=(const MaskExpr &other) const { return !equals(other); }

    std::shared_ptr<const Node> node;
  };

  struct BlockInfo;
  struct EdgeInfo;
  struct IfInfo;
  struct LoopInfo;
  struct SwitchInfo;

  MaskExpr materializeMaskExpr(mlir::Value &result, BlockInfo &current,
                               const MaskExpr &expr, mlir::OpBuilder &builder);

  enum class PayloadKind : uint8_t { Unknown, Result, Carried, Mask };

  struct EdgeInfo {
    BlockInfo *source = nullptr;
    BlockInfo *dest = nullptr;
    llvm::SmallVector<mlir::Value, 8> payload;
    llvm::SmallVector<PayloadKind, 8> payloadKinds;
    llvm::SmallVector<mlir::Value, 4> control;
    llvm::SmallVector<mlir::Value, 4> maskValues;
    MaskExpr guardMask;
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
    BlockInfo *mergeBlock = nullptr;
    mlir::Block *mergeOriginal = nullptr;
    llvm::SmallVector<mlir::BlockArgument, 4> mergeArgs;
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    llvm::SmallVector<mlir::Value, 4> results;
    mlir::Value condition;
    llvm::SmallVector<mlir::Value, 4> thenYieldValues;
    llvm::SmallVector<mlir::Value, 4> elseYieldValues;
    mlir::Operation *thenYieldOp = nullptr;
    mlir::Operation *elseYieldOp = nullptr;
    bool elseImplicitYield = false;
  };

  struct LoopInfo {
    mlir::Operation *op = nullptr;
    BlockInfo *parent = nullptr;
    BlockInfo *prepareBlock = nullptr;
    BlockInfo *bodyBlock = nullptr;
    BlockInfo *mergeBlock = nullptr;
    llvm::SmallVector<mlir::BlockArgument, 4> mergeArgs;
    llvm::SmallVector<mlir::Value, 4> forwardedToBody;
    llvm::SmallVector<mlir::Value, 4> forwardedToExit;
  };

  struct SwitchInfo {
    mlir::Operation *op = nullptr;
    BlockInfo *parent = nullptr;
    llvm::SmallVector<BlockInfo *, 4> caseBlocks;
    BlockInfo *defaultBlock = nullptr;
    llvm::SmallVector<int64_t, 8> caseValues;
    unsigned carriedCount = 0;
    unsigned payloadCount = 0;
    bool hasControlFlags = false;
    struct CaseRecord {
      BlockInfo *block = nullptr;
      BlockInfo *nextCase = nullptr;
      mlir::Value matchSeen;
      mlir::Value fallthrough;
      mlir::Value switchDone;
      llvm::SmallVector<mlir::Value, 4> carriedValues;
      llvm::SmallVector<mlir::Value, 4> payloadValues;
      llvm::SmallVector<mlir::Value, 4> controlValues;
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

    MaskExpr incomingMask;
    bool maskKnown = false;

    mlir::Operation *originalTerminator = nullptr;

    simt::structured::BlockOp structuredOp;
    mlir::Block *structuredBody = nullptr;
    mlir::BlockArgument structuredMaskArg;
    mlir::Value currentMask;
    llvm::SmallVector<mlir::BlockArgument, 4> structuredArgs;
    llvm::SmallVector<mlir::BlockArgument, 4> payloadArgs;
    llvm::SmallVector<PayloadKind, 8> payloadKinds;
    mlir::Operation *owningIf = nullptr;
    unsigned payloadBlockArgOffset = 0;

    mlir::Block *mergeTarget = nullptr;
    mlir::Block *continueTarget = nullptr;

    bool isMergeBlock = false;
    llvm::SmallVector<mlir::Value, 4> capturedInputs;
    llvm::SmallVector<mlir::BlockArgument, 4> capturedArgs;
    llvm::DenseMap<mlir::Value, unsigned> capturedInputIndex;

    bool requestsMaskPush = false;
    bool requestsMaskPop = false;
    std::string symbolName;
    llvm::SmallVector<unsigned, 4> outgoingEdges;
  };

  mlir::LogicalResult analyseBlocks();
  mlir::LogicalResult computeMasks();
  mlir::LogicalResult computePayloads();
  mlir::LogicalResult enumerateEdges();
  mlir::LogicalResult emitStructuredBlocks();
  mlir::LogicalResult cleanupOriginalCFG();

  mlir::LogicalResult emitStructuredBlock(BlockInfo &info);
  mlir::LogicalResult emitStructuredTerminator(BlockInfo &source);
  mlir::LogicalResult emitStructuredIf(BlockInfo &header, IfInfo &info,
                                       mlir::OpBuilder &builder);
  mlir::LogicalResult emitStructuredLoop(BlockInfo &enclosing, LoopInfo &info,
                                         mlir::OpBuilder &builder);
  void normalizeEdgeForMerge(EdgeInfo &edge, BlockInfo &merge);
  static mlir::Block *getSuccessorBody(const BlockInfo &succ);
  static unsigned getDataArgCount(const BlockInfo &succ);
  static mlir::BlockArgument getDataArgAt(const BlockInfo &succ,
                                          unsigned index);
  mlir::BlockArgument getCapturedArg(BlockInfo &succ, mlir::Value value);
  void appendCapturedInputs(EdgeInfo &edge);
  void computeCapturedInputs(BlockInfo &info);
  mlir::LogicalResult materializeEdgeOperands(EdgeInfo &edge, BlockInfo *succ,
                                              llvm::SmallVectorImpl<mlir::Value> &operands,
                                              mlir::Operation *context);
  mlir::LogicalResult stabilisePayloadSeeds();

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

  BlockInfo &ensureLoopMergeBlock(LoopInfo &info, mlir::Operation *loopOp);
  BlockInfo &ensureIfMergeBlock(IfInfo &info, mlir::Operation *ifOp);

  mlir::FunctionOpInterface func;
  llvm::SmallVector<mlir::Block *> blockOrder;

  /// Mapping from original blocks to collected metadata.
  llvm::DenseMap<mlir::Block *, BlockInfo> blockInfos;
  llvm::SmallVector<EdgeInfo, 4> edges;

  /// Scratch storage used while cloning ops into structured blocks.
  std::unique_ptr<mlir::IRMapping> mapper;
  std::unique_ptr<mlir::DominanceInfo> domInfo;

  llvm::DenseMap<mlir::Operation *, IfInfo> ifInfos;
  llvm::DenseMap<mlir::Operation *, LoopInfo> loopInfos;
  llvm::DenseMap<mlir::Operation *, SwitchInfo> switchInfos;

  llvm::SmallVector<mlir::Value, 4> functionReturnValues;
  bool hasFunctionReturn = false;

  llvm::SmallVector<simt::structured::BlockOp, 8> structuredOpsInOrder;
};

} // namespace conversion

#endif // SIMT_STEP_CONVERSION_STRUCTURED_CFGBUILDER_H
