#include "simt-hlsl-import/LoopScopeSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceManager.h"

namespace simt_hlsl_import {

namespace {

struct ContextDiagSink : DiagSink {
  explicit ContextDiagSink(LoweringContext &ctx) : ctx(ctx) {}

  void report(SourceLoc /*loc*/, llvm::StringRef message) override {
    if (!ctx.failed)
      ctx.errorMessage = message.str();
    ctx.failed = true;
  }

  LoweringContext &ctx;
};

bool reportContextError(LoweringContext &ctx, llvm::StringRef message) {
  ctx.diagnostics().report({nullptr, ctx.defaultLoc}, message);
  ctx.failed = true;
  return false;
}

} // namespace

SourceLoc makeSourceLoc(const clang::Stmt *stmt, LoweringContext &ctx) {
  if (!stmt)
    return SourceLoc{nullptr, ctx.defaultLoc};

  if (!ctx.sourceManager)
    return SourceLoc{stmt, ctx.defaultLoc};

  const clang::SourceManager &sm = *ctx.sourceManager;
  clang::SourceLocation loc = stmt->getBeginLoc();
  if (loc.isInvalid())
    loc = stmt->getEndLoc();
  if (loc.isInvalid())
    return SourceLoc{stmt, ctx.defaultLoc};

  loc = sm.getExpansionLoc(loc);
  clang::PresumedLoc presumed = sm.getPresumedLoc(loc);
  if (!presumed.isValid())
    return SourceLoc{stmt, ctx.defaultLoc};

  mlir::MLIRContext *mlirCtx = ctx.builder.getContext();
  mlir::StringAttr fileAttr =
      mlir::StringAttr::get(mlirCtx, presumed.getFilename());
  mlir::Location mlirLoc =
      mlir::FileLineColLoc::get(fileAttr, presumed.getLine(),
                                presumed.getColumn());
  return SourceLoc{stmt, mlirLoc};
}

SourceLoc makeSourceLoc(const clang::Decl *decl, LoweringContext &ctx) {
  if (!decl)
    return SourceLoc{nullptr, ctx.defaultLoc};

  if (!ctx.sourceManager)
    return SourceLoc{nullptr, ctx.defaultLoc};

  const clang::SourceManager &sm = *ctx.sourceManager;
  clang::SourceLocation loc = decl->getLocation();
  if (loc.isInvalid())
    return SourceLoc{nullptr, ctx.defaultLoc};

  loc = sm.getExpansionLoc(loc);
  clang::PresumedLoc presumed = sm.getPresumedLoc(loc);
  if (!presumed.isValid())
    return SourceLoc{nullptr, ctx.defaultLoc};

  mlir::MLIRContext *mlirCtx = ctx.builder.getContext();
  mlir::StringAttr fileAttr =
      mlir::StringAttr::get(mlirCtx, presumed.getFilename());
  mlir::Location mlirLoc =
      mlir::FileLineColLoc::get(fileAttr, presumed.getLine(),
                                presumed.getColumn());
  return SourceLoc{nullptr, mlirLoc};
}

bool isEmitContext(const LoweringContext &ctx) {
  mlir::Block *block = ctx.builder.getInsertionBlock();
  return block && block->getParentOp();
}

SymValue makeSymValue(const clang::ValueDecl *decl) {
  SymValue sym;
  if (!decl)
    return sym;
  clang::QualType qt = decl->getType();
  if (qt.isNull())
    return sym;
  sym.isConst = qt.isConstQualified();
  const clang::Type *type = qt.getTypePtrOrNull();
  if (!type)
    return sym;
  clang::ASTContext &astCtx = decl->getASTContext();
  if (type->isBooleanType() || type->isIntegerType()) {
    sym.kind = SymKind::ScalarInt;
    clang::TypeInfo info = astCtx.getTypeInfo(qt);
    sym.bitWidth = static_cast<unsigned>(info.Width);
  } else if (type->isFloatingType()) {
    sym.kind = SymKind::ScalarFloat;
    clang::TypeInfo info = astCtx.getTypeInfo(qt);
    sym.bitWidth = static_cast<unsigned>(info.Width);
  } else if (type->isVectorType()) {
    sym.kind = SymKind::Vector;
    if (const auto *vecTy = type->getAs<clang::VectorType>()) {
      sym.elementCount = vecTy->getNumElements();
      clang::TypeInfo info = astCtx.getTypeInfo(vecTy->getElementType());
      sym.bitWidth = static_cast<unsigned>(info.Width);
    }
  } else if (type->isPointerType()) {
    sym.kind = SymKind::Pointer;
    sym.bitWidth = static_cast<unsigned>(astCtx.getTypeSize(qt));
  }
  return sym;
}

mlir::Value getLoopCarriedValue(const LoweringContext &ctx,
                                const clang::ValueDecl *vd) {
  auto it = ctx.valueMap.find(vd);
  if (it != ctx.valueMap.end())
    return it->second;
  return {};
}

namespace {

void collectLoopOperandsRaw(LoweringContext &ctx, LoopFrame &frame,
                            llvm::SmallVectorImpl<mlir::Value> &ops,
                            bool /*isContinue*/) {
  ops.clear();
  if (!frame.loop)
    return;
  mlir::Block &bodyBlock = frame.loop.getBodyRegion().front();
  ops.reserve(frame.carriedVars.size() + (frame.hasFirstIterFlag ? 1 : 0));
  for (auto [index, vd] : llvm::enumerate(frame.carriedVars)) {
    mlir::Value value = ctx.valueMap.lookup(vd);
    if (!value && index < bodyBlock.getNumArguments())
      value = bodyBlock.getArgument(index);
    if (!value)
      value = frame.loop.getResult(index);
    ops.push_back(value);
  }
  if (frame.hasFirstIterFlag) {
    mlir::Value flag = frame.currentFirstIterValue;
    if (!flag && frame.firstIterIndex < frame.loop.getNumResults())
      flag = frame.loop.getResult(frame.firstIterIndex);
    ops.push_back(flag);
  }
}

void collectLoopExitOperandsConcrete(LoweringContext &ctx, LoopFrame &frame,
                                     bool isContinue,
                                     llvm::SmallVectorImpl<mlir::Value> &ops) {
  llvm::SmallVector<mlir::Value, 8> allOperands;
  collectLoopOperandsRaw(ctx, frame, allOperands, isContinue);

  unsigned carriedCount = frame.carriedVars.size();
  LoopMetadata *metadata = frame.metadata;

  bool anySelected = false;
  if (metadata) {
    for (unsigned index = 0; index < carriedCount; ++index) {
      const clang::ValueDecl *vd = frame.carriedVars[index];
      bool forward = false;
      if (auto it = metadata->mutationInfo.find(vd); it != metadata->mutationInfo.end())
        forward = isContinue ? it->second.mutatedOnContinue
                             : it->second.mutatedOnBreak;
      if (forward) {
        ops.push_back(allOperands[index]);
        anySelected = true;
      }
    }
  }

  if (!metadata || !anySelected) {
    for (unsigned index = 0; index < carriedCount; ++index)
      ops.push_back(allOperands[index]);
  }

  if (frame.hasFirstIterFlag && allOperands.size() > carriedCount)
    ops.push_back(allOperands.back());
}

} // namespace

void collectLoopBreakOperands(LoweringContext &ctx, LoopFrame &frame,
                              llvm::SmallVectorImpl<mlir::Value> &ops) {
  if (frame.analysisOnly) {
    ops.clear();
    return;
  }
  if (frame.activeScope)
    return frame.activeScope->collectBreakOperands(ctx, ops);
  collectLoopExitOperandsConcrete(ctx, frame, /*isContinue=*/false, ops);
}

void collectLoopContinueOperands(LoweringContext &ctx, LoopFrame &frame,
                                 llvm::SmallVectorImpl<mlir::Value> &ops) {
  if (frame.analysisOnly) {
    ops.clear();
    return;
  }
  if (frame.activeScope)
    return frame.activeScope->collectContinueOperands(ctx, ops);
  collectLoopExitOperandsConcrete(ctx, frame, /*isContinue=*/true, ops);
}

LoopFrame *getInnermostLoop(LoweringContext &ctx) {
  for (auto it = ctx.controlStack.rbegin(); it != ctx.controlStack.rend(); ++it) {
    if (it->kind != ControlEntryKind::Loop)
      continue;
    if (it->index >= ctx.loopStack.size())
      continue;
    return &ctx.loopStack[it->index];
  }
  return nullptr;
}

BreakTarget getInnermostBreakTarget(LoweringContext &ctx) {
  for (auto it = ctx.controlStack.rbegin(); it != ctx.controlStack.rend(); ++it) {
    if (it->kind == ControlEntryKind::Loop) {
      if (it->index >= ctx.loopStack.size())
        continue;
      return BreakTarget{ControlEntryKind::Loop, &ctx.loopStack[it->index],
                         nullptr};
    }
    if (it->kind == ControlEntryKind::Switch) {
      if (it->index >= ctx.switchStack.size())
        continue;
      return BreakTarget{ControlEntryKind::Switch, nullptr,
                         &ctx.switchStack[it->index]};
    }
  }
  return BreakTarget{};
}

SwitchScopeGuard::SwitchScopeGuard(LoweringContext &ctx, SwitchFrame frame)
    : ctx(&ctx), valid(true) {
  ctx.switchStack.push_back(std::move(frame));
  ctx.controlStack.push_back(
      {ControlEntryKind::Switch, ctx.switchStack.size() - 1});
  if (!ctx.switchMetadataStack.empty())
    ctx.switchStack.back().metadata = ctx.switchMetadataStack.back();
}

SwitchScopeGuard::~SwitchScopeGuard() {
  if (!valid || !ctx)
    return;
  assert(!ctx->switchStack.empty() &&
         "SwitchScopeGuard outlived switch stack entry");
  ctx->switchStack.pop_back();
  if (!ctx->controlStack.empty() &&
      ctx->controlStack.back().kind == ControlEntryKind::Switch)
    ctx->controlStack.pop_back();
}

SwitchFrame &SwitchScopeGuard::frame() {
  assert(ctx && !ctx->switchStack.empty());
  return ctx->switchStack.back();
}

void cloneContextState(const LoweringContext &parent, LoweringContext &child) {
  child.valueMap = parent.valueMap;
  child.symValueMap = parent.symValueMap;
  child.loopStack = parent.loopStack;
  child.switchStack = parent.switchStack;
  child.controlStack = parent.controlStack;
  child.loopMetadataStack = parent.loopMetadataStack;
  child.switchMetadataStack = parent.switchMetadataStack;
  child.sourceManager = parent.sourceManager;
  child.astContext = parent.astContext;
  child.diagSink = parent.diagSink;
  child.ownedDiagSink.reset();
}

SwitchFrame makeSwitchFrame(LoweringContext &ctx,
                            llvm::ArrayRef<const clang::ValueDecl *> carriedVars,
                            llvm::ArrayRef<mlir::Value> currentValues,
                            mlir::Location loc,
                            bool hasMatchedDefault, bool executingDefault,
                            bool completedDefault) {
  SwitchFrame frame;
  frame.carriedVars.append(carriedVars.begin(), carriedVars.end());
  frame.hasMatchedIndex = carriedVars.size();
  frame.executingIndex = carriedVars.size() + 1;
  frame.completedIndex = carriedVars.size() + 2;
  frame.initialValues.assign(currentValues.begin(), currentValues.end());
  frame.breakHasMatchedValue =
      ctx.builder.create<mlir::arith::ConstantIntOp>(loc, hasMatchedDefault, 1);
  frame.breakExecutingValue =
      ctx.builder.create<mlir::arith::ConstantIntOp>(loc, executingDefault, 1);
  frame.breakCompletedValue =
      ctx.builder.create<mlir::arith::ConstantIntOp>(loc, completedDefault, 1);
  if (!ctx.switchMetadataStack.empty())
    frame.metadata = ctx.switchMetadataStack.back();
  return frame;
}

bool buildLoopSkeleton(LoweringContext &ctx,
                       llvm::ArrayRef<const clang::ValueDecl *> mutatedVars,
                       bool hasFirstIterFlag, mlir::Value firstIterInit,
                       LoopSkeleton &out) {
  mlir::Location loc = ctx.defaultLoc;

  llvm::SmallVector<mlir::Type, 8> resultTypes;
  llvm::SmallVector<mlir::Value, 8> initValues;
  resultTypes.reserve(mutatedVars.size() + (hasFirstIterFlag ? 1 : 0));
  initValues.reserve(mutatedVars.size() + (hasFirstIterFlag ? 1 : 0));

  for (const clang::ValueDecl *vd : mutatedVars) {
    mlir::Value initial = getLoopCarriedValue(ctx, vd);
    if (!initial)
      return reportContextError(ctx, "reference to unknown loop variable");
    resultTypes.push_back(initial.getType());
    initValues.push_back(initial);
  }

  if (hasFirstIterFlag) {
    if (!firstIterInit)
      firstIterInit = ctx.builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    resultTypes.push_back(ctx.builder.getI1Type());
    initValues.push_back(firstIterInit);
  }

  auto loop =
      ctx.builder.create<simt::dialect::LoopOp>(loc, resultTypes, initValues);

  LoopFrame frame{loop, {}, /*hasFirstIterFlag=*/false,
                  /*firstIterIndex=*/0, /*currentFirstIterValue=*/mlir::Value()};
  frame.analysisOnly = !isEmitContext(ctx);
  frame.carriedVars.append(mutatedVars.begin(), mutatedVars.end());
  frame.hasFirstIterFlag = hasFirstIterFlag;
  if (hasFirstIterFlag) {
    frame.firstIterIndex = mutatedVars.size();
    frame.currentFirstIterValue = firstIterInit;
  }
  if (!ctx.loopMetadataStack.empty())
    frame.metadata = ctx.loopMetadataStack.back();
  ctx.loopStack.push_back(frame);
  ctx.controlStack.push_back({ControlEntryKind::Loop, ctx.loopStack.size() - 1});
  LoopFrame *activeFrame = &ctx.loopStack.back();

  auto &prepareRegion = loop.getPrepareRegion();
  if (prepareRegion.empty())
    prepareRegion.emplaceBlock();
  auto *prepareBlock = &prepareRegion.front();
  if (!resultTypes.empty()) {
    llvm::SmallVector<mlir::Location, 8> argLocs(resultTypes.size(), loc);
    prepareBlock->addArguments(resultTypes, argLocs);
  }

  auto &bodyRegion = loop.getBodyRegion();
  if (bodyRegion.empty())
    bodyRegion.emplaceBlock();
  auto *bodyBlock = &bodyRegion.front();
  if (!resultTypes.empty()) {
    llvm::SmallVector<mlir::Location, 8> argLocs(resultTypes.size(), loc);
    bodyBlock->addArguments(resultTypes, argLocs);
  }

  out.loop = loop;
  out.frame = activeFrame;
  out.prepareBlock = prepareBlock;
  out.bodyBlock = bodyBlock;

  return true;
}

LoopScopeState::LoopScopeState(LoweringContext &parentCtx,
                               llvm::ArrayRef<const clang::ValueDecl *> carried,
                               bool hasFirstIterFlag, mlir::Value firstIterInit,
                               mlir::Location loc)
    : parent(parentCtx), loc(loc) {
  carriedVars.assign(carried.begin(), carried.end());
  if (!buildLoopSkeleton(parent, carriedVars, hasFirstIterFlag, firstIterInit,
                         skeleton)) {
    active = false;
    return;
  }
  active = true;
  if (skeleton.frame)
    skeleton.frame->activeScope = this;
  setupPrepareContext();
  setupBodyContext();
}

LoopScopeState::~LoopScopeState() { cleanup(); }

bool LoopScopeState::isValid() const { return active && !parent.failed; }

LoweringContext &LoopScopeState::prepareContext() {
  assert(active && prepareCtx && "prepare context unavailable");
  return *prepareCtx;
}

LoweringContext &LoopScopeState::bodyContext() {
  assert(active && bodyCtx && "body context unavailable");
  return *bodyCtx;
}

bool LoopScopeState::isAnalysisOnly() const {
  return skeleton.frame && skeleton.frame->analysisOnly;
}

bool LoopScopeState::hasFirstIterFlag() const {
  return skeleton.frame && skeleton.frame->hasFirstIterFlag;
}

unsigned LoopScopeState::getFirstIterIndex() const {
  assert(hasFirstIterFlag());
  return skeleton.frame->firstIterIndex;
}

mlir::Value LoopScopeState::getPrepareFirstIterArg() const {
  assert(hasFirstIterFlag());
  return skeleton.prepareBlock->getArgument(getFirstIterIndex());
}

mlir::Value LoopScopeState::getBodyFirstIterArg() const {
  assert(hasFirstIterFlag());
  return skeleton.bodyBlock->getArgument(getFirstIterIndex());
}

void LoopScopeState::setCurrentFirstIterValue(mlir::Value value) {
  if (!hasFirstIterFlag())
    return;
  skeleton.frame->currentFirstIterValue = value;
  if (prepareCtx && !prepareCtx->loopStack.empty())
    prepareCtx->loopStack.back().currentFirstIterValue = value;
  if (bodyCtx && !bodyCtx->loopStack.empty())
    bodyCtx->loopStack.back().currentFirstIterValue = value;
}

bool LoopScopeState::close() {
  if (!active)
    return !parent.failed;

  if (skeleton.frame && skeleton.frame->analysisOnly) {
    if (bodyCtx) {
      parent.failed |= bodyCtx->failed;
      parent.mutatedVars.insert(bodyCtx->mutatedVars.begin(),
                                bodyCtx->mutatedVars.end());
      for (const clang::ValueDecl *vd : carriedVars) {
        if (auto it = bodyCtx->valueMap.find(vd);
            it != bodyCtx->valueMap.end())
          parent.valueMap[vd] = it->second;
        if (auto symIt = bodyCtx->symValueMap.find(vd);
            symIt != bodyCtx->symValueMap.end())
          parent.symValueMap[vd] = symIt->second;
      }
    }
    parent.mutatedVars.insert(carriedVars.begin(), carriedVars.end());
    cleanup();
    return !parent.failed;
  }

  unsigned index = 0;
  for (const clang::ValueDecl *vd : carriedVars) {
    parent.valueMap[vd] = skeleton.loop.getResult(index++);
    parent.mutatedVars.insert(vd);
    if (bodyCtx) {
      if (auto it = bodyCtx->symValueMap.find(vd);
          it != bodyCtx->symValueMap.end())
        parent.symValueMap[vd] = it->second;
    }
  }

  if (bodyCtx) {
    parent.failed |= bodyCtx->failed;
    parent.mutatedVars.insert(bodyCtx->mutatedVars.begin(),
                               bodyCtx->mutatedVars.end());
  }

  cleanup();
  return !parent.failed;
}

void LoopScopeState::collectBreakOperands(
    LoweringContext &ctx, llvm::SmallVectorImpl<mlir::Value> &ops) {
  if (skeleton.frame && skeleton.frame->analysisOnly) {
    ops.clear();
    return;
  }
  if (skeleton.frame)
    collectLoopExitOperandsConcrete(ctx, *skeleton.frame,
                                    /*isContinue=*/false, ops);
}

void LoopScopeState::collectContinueOperands(
    LoweringContext &ctx, llvm::SmallVectorImpl<mlir::Value> &ops) {
  if (skeleton.frame && skeleton.frame->analysisOnly) {
    ops.clear();
    return;
  }
  if (skeleton.frame)
    collectLoopExitOperandsConcrete(ctx, *skeleton.frame,
                                    /*isContinue=*/true, ops);
}

void LoopScopeState::setupPrepareContext() {
  prepareBuilder.emplace(parent.builder.getContext());
  prepareBuilder->setInsertionPointToStart(skeleton.prepareBlock);
  prepareCtx = std::make_unique<LoweringContext>(
      *prepareBuilder, loc, parent.returnType, parent.errorMessage,
      parent.sourceManager, parent.astContext);
  copySharedState(*prepareCtx);
  setBlockArguments(*prepareCtx, skeleton.prepareBlock);
}

void LoopScopeState::setupBodyContext() {
  bodyBuilder.emplace(parent.builder.getContext());
  bodyBuilder->setInsertionPointToStart(skeleton.bodyBlock);
  bodyCtx = std::make_unique<LoweringContext>(
      *bodyBuilder, loc, parent.returnType, parent.errorMessage,
      parent.sourceManager, parent.astContext);
  copySharedState(*bodyCtx);
  setBlockArguments(*bodyCtx, skeleton.bodyBlock);
}

void LoopScopeState::copySharedState(LoweringContext &childCtx) {
  cloneContextState(parent, childCtx);
}

void LoopScopeState::setBlockArguments(LoweringContext &childCtx,
                                       mlir::Block *block) {
  if (!block)
    return;
  for (auto [idx, vd] : llvm::enumerate(carriedVars)) {
    if (idx < block->getNumArguments())
      childCtx.valueMap[vd] = block->getArgument(idx);
  }
  if (hasFirstIterFlag() && !childCtx.loopStack.empty() && skeleton.frame &&
      skeleton.frame->hasFirstIterFlag &&
      skeleton.frame->firstIterIndex < block->getNumArguments()) {
    childCtx.loopStack.back().currentFirstIterValue =
        block->getArgument(skeleton.frame->firstIterIndex);
  }
}

void LoopScopeState::cleanup() {
  if (!active)
    return;
  active = false;
  if (skeleton.frame && skeleton.frame->activeScope == this)
    skeleton.frame->activeScope = nullptr;
  if (!parent.loopStack.empty())
    parent.loopStack.pop_back();
  if (!parent.controlStack.empty() &&
      parent.controlStack.back().kind == ControlEntryKind::Loop)
    parent.controlStack.pop_back();
}

AnalysisLoopScope::AnalysisLoopScope(
    LoweringContext &parentCtx,
    llvm::ArrayRef<const clang::ValueDecl *> carried,
    bool hasFirstIterFlag, mlir::Value firstIterInit, mlir::Location loc)
    : parent(parentCtx), loc(loc) {
  carriedVars.assign(carried.begin(), carried.end());

  llvm::SmallVector<mlir::Type, 8> argTypes;
  argTypes.reserve(carriedVars.size() + (hasFirstIterFlag ? 1 : 0));

  for (const clang::ValueDecl *vd : carriedVars) {
    mlir::Value initial = getLoopCarriedValue(parent, vd);
    if (!initial) {
      parent.fail("reference to unknown loop variable");
      return;
    }
    argTypes.push_back(initial.getType());
  }

  if (hasFirstIterFlag)
    argTypes.push_back(parent.builder.getI1Type());

  if (parent.failed)
    return;

  LoopFrame frameState{};
  frameState.carriedVars.append(carriedVars.begin(), carriedVars.end());
  frameState.analysisOnly = true;
  frameState.hasFirstIterFlag = hasFirstIterFlag;
  if (hasFirstIterFlag) {
    frameState.firstIterIndex = carriedVars.size();
    frameState.currentFirstIterValue = firstIterInit;
  }
  if (!parent.loopMetadataStack.empty())
    frameState.metadata = parent.loopMetadataStack.back();

  parent.loopStack.push_back(frameState);
  parent.controlStack.push_back({ControlEntryKind::Loop,
                                 parent.loopStack.size() - 1});
  frame = &parent.loopStack.back();
  frame->activeScope = this;
  if (hasFirstIterFlag)
    frame->currentFirstIterValue = firstIterInit;

  active = true;

  prepareBlock = &prepareRegion.emplaceBlock();
  bodyBlock = &bodyRegion.emplaceBlock();
  if (!argTypes.empty()) {
    llvm::SmallVector<mlir::Location, 8> argLocs(argTypes.size(), loc);
    prepareBlock->addArguments(argTypes, argLocs);
    bodyBlock->addArguments(argTypes, argLocs);
  }

  setupPrepareContext();
  setupBodyContext();
}

AnalysisLoopScope::~AnalysisLoopScope() { cleanup(); }

bool AnalysisLoopScope::isValid() const { return active && !parent.failed; }

LoweringContext &AnalysisLoopScope::prepareContext() {
  assert(active && prepareCtx && "prepare context unavailable");
  return *prepareCtx;
}

LoweringContext &AnalysisLoopScope::bodyContext() {
  assert(active && bodyCtx && "body context unavailable");
  return *bodyCtx;
}

bool AnalysisLoopScope::isAnalysisOnly() const { return true; }

bool AnalysisLoopScope::hasFirstIterFlag() const {
  return frame && frame->hasFirstIterFlag;
}

unsigned AnalysisLoopScope::getFirstIterIndex() const {
  assert(hasFirstIterFlag());
  return frame->firstIterIndex;
}

mlir::Value AnalysisLoopScope::getPrepareFirstIterArg() const {
  assert(hasFirstIterFlag() && prepareBlock);
  return prepareBlock->getArgument(getFirstIterIndex());
}

mlir::Value AnalysisLoopScope::getBodyFirstIterArg() const {
  assert(hasFirstIterFlag() && bodyBlock);
  return bodyBlock->getArgument(getFirstIterIndex());
}

void AnalysisLoopScope::setCurrentFirstIterValue(mlir::Value value) {
  if (!hasFirstIterFlag())
    return;
  frame->currentFirstIterValue = value;
  if (prepareCtx && !prepareCtx->loopStack.empty())
    prepareCtx->loopStack.back().currentFirstIterValue = value;
  if (bodyCtx && !bodyCtx->loopStack.empty())
    bodyCtx->loopStack.back().currentFirstIterValue = value;
}

bool AnalysisLoopScope::close() {
  if (!active)
    return !parent.failed;

  if (bodyCtx) {
    parent.failed |= bodyCtx->failed;
    parent.mutatedVars.insert(bodyCtx->mutatedVars.begin(),
                               bodyCtx->mutatedVars.end());
    for (const clang::ValueDecl *vd : carriedVars) {
      if (auto it = bodyCtx->valueMap.find(vd);
          it != bodyCtx->valueMap.end())
        parent.valueMap[vd] = it->second;
      if (auto symIt = bodyCtx->symValueMap.find(vd);
          symIt != bodyCtx->symValueMap.end())
        parent.symValueMap[vd] = symIt->second;
    }
  }

  parent.mutatedVars.insert(carriedVars.begin(), carriedVars.end());

  cleanup();
  return !parent.failed;
}

void AnalysisLoopScope::collectBreakOperands(
    LoweringContext &, llvm::SmallVectorImpl<mlir::Value> &ops) {
  ops.clear();
}

void AnalysisLoopScope::collectContinueOperands(
    LoweringContext &, llvm::SmallVectorImpl<mlir::Value> &ops) {
  ops.clear();
}

DiagSink &LoweringContext::diagnostics() {
  if (diagSink)
    return *diagSink;
  ownedDiagSink = std::make_unique<ContextDiagSink>(*this);
  diagSink = ownedDiagSink.get();
  return *diagSink;
}

void AnalysisLoopScope::setupPrepareContext() {
  prepareBuilder.emplace(parent.builder.getContext());
  prepareBuilder->setInsertionPointToStart(prepareBlock);
  prepareCtx = std::make_unique<LoweringContext>(
      *prepareBuilder, loc, parent.returnType, parent.errorMessage,
      parent.sourceManager, parent.astContext);
  copySharedState(*prepareCtx);
  setBlockArguments(*prepareCtx, prepareBlock);
}

void AnalysisLoopScope::setupBodyContext() {
  bodyBuilder.emplace(parent.builder.getContext());
  bodyBuilder->setInsertionPointToStart(bodyBlock);
  bodyCtx = std::make_unique<LoweringContext>(
      *bodyBuilder, loc, parent.returnType, parent.errorMessage,
      parent.sourceManager, parent.astContext);
  copySharedState(*bodyCtx);
  setBlockArguments(*bodyCtx, bodyBlock);
}

void AnalysisLoopScope::copySharedState(LoweringContext &childCtx) {
  cloneContextState(parent, childCtx);
}

void AnalysisLoopScope::setBlockArguments(LoweringContext &childCtx,
                                          mlir::Block *block) {
  if (!block)
    return;
  for (auto [idx, vd] : llvm::enumerate(carriedVars)) {
    if (idx < block->getNumArguments())
      childCtx.valueMap[vd] = block->getArgument(idx);
  }
  if (hasFirstIterFlag() && !childCtx.loopStack.empty() && frame &&
      frame->firstIterIndex < block->getNumArguments()) {
    childCtx.loopStack.back().currentFirstIterValue =
        block->getArgument(frame->firstIterIndex);
  }
}

void AnalysisLoopScope::cleanup() {
  if (!active) {
    prepareCtx.reset();
    bodyCtx.reset();
    prepareBuilder.reset();
    bodyBuilder.reset();
    return;
  }

  active = false;
  if (frame && frame->activeScope == this)
    frame->activeScope = nullptr;
  if (!parent.loopStack.empty())
    parent.loopStack.pop_back();
  if (!parent.controlStack.empty() &&
      parent.controlStack.back().kind == ControlEntryKind::Loop)
    parent.controlStack.pop_back();
  frame = nullptr;
  prepareCtx.reset();
  bodyCtx.reset();
  prepareBuilder.reset();
  bodyBuilder.reset();
}

} // namespace simt_hlsl_import
