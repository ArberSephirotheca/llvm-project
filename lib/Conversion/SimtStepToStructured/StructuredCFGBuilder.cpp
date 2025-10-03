#include "simt-step/Conversion/StructuredCFGBuilder.h"

#include "simt-step/Dialect/SimtStructured/StructuredDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"

#include <llvm/ADT/StringRef.h>

using namespace mlir;

namespace simt::conversion {

namespace {

constexpr llvm::StringLiteral kUnimplementedMsg(
    "StructuredCFGBuilder skeleton reached. Implement the new builder.");

static LogicalResult signalUnimplemented(FunctionOpInterface func) {
  func.emitError(kUnimplementedMsg);
  return failure();
}

} // namespace

struct StructuredCFGBuilder::BlockInfo {
  mlir::Block *original = nullptr;
  SmallVector<mlir::Type, 4> carriedTypes;
  SmallVector<mlir::Value, 4> payloadSeed;
  SmallVector<mlir::BlockArgument, 4> blockArgs;

  simt::structured::BlockOp structuredOp;
  mlir::Block *structuredBody = nullptr;

  mlir::Block *mergeTarget = nullptr;
  mlir::Block *continueTarget = nullptr;

  bool requestsMaskPush = false;
  bool requestsMaskPop = false;
};

struct StructuredCFGBuilder::EdgeInfo {
  BlockInfo *source = nullptr;
  BlockInfo *dest = nullptr;
  SmallVector<mlir::Value, 8> payload;
  SmallVector<mlir::Value, 4> maskValues;
  enum Kind { Plain, ConditionalTrue, ConditionalFalse, LoopBackEdge } kind =
      Plain;
};

struct StructuredCFGBuilder::SwitchCaseInfo {
  BlockInfo *header = nullptr;
  SmallVector<BlockInfo *, 4> cases;
  BlockInfo *defaultCase = nullptr;
};

StructuredCFGBuilder::StructuredCFGBuilder(FunctionOpInterface func)
    : func(func) {}

LogicalResult StructuredCFGBuilder::build() {
  mapper = std::make_unique<IRMapping>();
  domInfo = std::make_unique<DominanceInfo>(func);

  collectOriginalBlocks();

  if (blockOrder.empty())
    return success();

  // TODO: implement the staged builder pipeline described in
  // docs/structured_cfg_builder_plan.md once analysis and payload propagation
  // logic lands.
  if (failed(analyseBlocks()))
    return failure();
  if (failed(computePayloads()))
    return failure();
  if (failed(enumerateEdges()))
    return failure();
  if (failed(emitStructuredBlocks()))
    return failure();
  if (failed(cleanupOriginalCFG()))
    return failure();

  return success();
}

void StructuredCFGBuilder::collectOriginalBlocks() {
  blockOrder.clear();
  for (Block &block : func.getBlocks())
    blockOrder.push_back(&block);
}

StructuredCFGBuilder::BlockInfo *
StructuredCFGBuilder::lookupBlockInfo(mlir::Block *block) {
  if (auto it = blockInfos.find(block); it != blockInfos.end())
    return &it->second;
  return nullptr;
}

const StructuredCFGBuilder::BlockInfo *
StructuredCFGBuilder::lookupBlockInfo(mlir::Block *block) const {
  if (auto it = blockInfos.find(block); it != blockInfos.end())
    return &it->second;
  return nullptr;
}

LogicalResult StructuredCFGBuilder::analyseBlocks() {
  for (mlir::Block *block : blockOrder) {
    BlockInfo &info = blockInfos[block];
    info.original = block;
    info.carriedTypes.reserve(block->getNumArguments());
    for (mlir::BlockArgument arg : block->getArguments())
      info.carriedTypes.push_back(arg.getType());

    // TODO: inspect structured ops inside the block and record merge/continue
    // intent.
  }

  // TODO: walk structured control ops (`simt.if`, `simt.loop`, `simt.switch`)
  // so BlockInfo captures per-header metadata for payload seeding.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::computePayloads() {
  // TODO: seed loop/switch headers with their initial payloads and iterate to
  // a fixed point so every block knows the tuple it expects.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::enumerateEdges() {
  // TODO: build one EdgeInfo per terminator using the payload map assembled by
  // computePayloads().
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::emitStructuredBlocks() {
  // TODO: create `simt_struct.block` ops mirroring `blockOrder`, clone bodies,
  // and materialise mask stack management before forwarding payloads.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::cleanupOriginalCFG() {
  // TODO: erase the legacy CFG blocks once the structured region is fully
  // populated.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::emitStructuredBlock(BlockInfo &info) {
  (void)info;
  // TODO: populate a single structured block, including carried arguments and
  // mask entry prologue.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::emitStructuredTerminator(
    BlockInfo &source, const EdgeInfo &edge) {
  (void)source;
  (void)edge;
  // TODO: create `branch`/`cond_branch` terminators with the payload tuple.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::analyseIfOp(BlockInfo &header,
                                                 mlir::Operation *op) {
  (void)header;
  (void)op;
  // TODO: capture `simt.if` merge target, else payloads, and record edges.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::analyseLoopOp(BlockInfo &header,
                                                   mlir::Operation *op) {
  (void)header;
  (void)op;
  // TODO: capture loop init payload, continue target, and carried values.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::analyseSwitchOp(BlockInfo &header,
                                                     mlir::Operation *op) {
  (void)header;
  (void)op;
  // TODO: gather per-case payloads, fallthrough info, and default edge.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::ensurePayloadShape(EdgeInfo &edge) {
  (void)edge;
  // TODO: expand operand tuples so that every edge forwards the correct arity.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::propagatePayload(
    BlockInfo &source, BlockInfo &dest, llvm::ArrayRef<mlir::Value> values) {
  (void)source;
  (void)dest;
  (void)values;
  // TODO: record the payload values reaching `dest` from `source`.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::materialiseMaskEntry(BlockInfo &info) {
  (void)info;
  // TODO: insert `mask_push`/`mask_merge` ops for the structured block.
  return signalUnimplemented(func);
}

LogicalResult StructuredCFGBuilder::materialiseMaskExit(BlockInfo &info) {
  (void)info;
  // TODO: add `mask_pop` when a block represents a merge point.
  return signalUnimplemented(func);
}

} // namespace simt::conversion
