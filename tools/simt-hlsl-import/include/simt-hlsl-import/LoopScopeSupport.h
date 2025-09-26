#pragma once

#include <memory>
#include <optional>
#include <string>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace clang {
class SourceManager;
class Stmt;
class ValueDecl;
} // namespace clang

namespace simt_hlsl_import {

struct SourceLoc;

enum class SymKind { Unknown, ScalarInt, ScalarFloat, Vector, Pointer };

struct SymValue {
  SymKind kind = SymKind::Unknown;
  unsigned bitWidth = 0;
  unsigned elementCount = 1;
  bool isConst = false;
};

struct LoweringContext;

struct LoopScopeProvider {
  virtual ~LoopScopeProvider() = default;
  virtual void collectBreakOperands(LoweringContext &ctx,
                                    llvm::SmallVectorImpl<mlir::Value> &ops) = 0;
  virtual void collectContinueOperands(
      LoweringContext &ctx, llvm::SmallVectorImpl<mlir::Value> &ops) = 0;
};

enum class ControlEntryKind { Loop, Switch };

struct LoopFrame {
  simt::dialect::LoopOp loop;
  llvm::SmallVector<const clang::ValueDecl *, 8> carriedVars;
  bool hasFirstIterFlag = false;
  unsigned firstIterIndex = 0;
  mlir::Value currentFirstIterValue;
  LoopScopeProvider *activeScope = nullptr;
};

struct LoopSkeleton {
  simt::dialect::LoopOp loop;
  LoopFrame *frame = nullptr;
  mlir::Block *prepareBlock = nullptr;
  mlir::Block *bodyBlock = nullptr;
};

struct SwitchFrame {
  llvm::SmallVector<const clang::ValueDecl *, 8> carriedVars;
  unsigned hasMatchedIndex = 0;
  unsigned executingIndex = 0;
  unsigned completedIndex = 0;
  llvm::SmallVector<mlir::Value, 8> initialValues;
  mlir::Value breakHasMatchedValue;
  mlir::Value breakExecutingValue;
  mlir::Value breakCompletedValue;
  bool analysisOnly = false;
};

struct ControlEntry {
  ControlEntryKind kind;
  size_t index;
};

struct LoweringContext {
  mlir::OpBuilder &builder;
  mlir::Location defaultLoc;
  mlir::Type returnType;
  llvm::DenseMap<const clang::ValueDecl *, mlir::Value> valueMap;
  llvm::DenseMap<const clang::ValueDecl *, SymValue> symValueMap;
  llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutatedVars;
  bool emittedTerminator = false;
  std::string &errorMessage;
  bool failed = false;
  llvm::SmallVector<LoopFrame, 4> loopStack;
  llvm::SmallVector<SwitchFrame, 4> switchStack;
  llvm::SmallVector<ControlEntry, 8> controlStack;
  const clang::SourceManager *sourceManager = nullptr;

  LoweringContext(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Type retType, std::string &error,
                  const clang::SourceManager *sm = nullptr)
      : builder(builder), defaultLoc(loc), returnType(retType),
        errorMessage(error), sourceManager(sm) {}

  bool fail(llvm::StringRef msg) {
    if (!failed)
      errorMessage = msg.str();
    failed = true;
    return false;
  }
};

bool isEmitContext(const LoweringContext &ctx);

SymValue makeSymValue(const clang::ValueDecl *decl);

mlir::Value getLoopCarriedValue(const LoweringContext &ctx,
                                const clang::ValueDecl *vd);

bool buildLoopSkeleton(LoweringContext &ctx,
                       llvm::ArrayRef<const clang::ValueDecl *> mutatedVars,
                       bool hasFirstIterFlag, mlir::Value firstIterInit,
                       LoopSkeleton &out);

void collectLoopBreakOperands(LoweringContext &ctx, LoopFrame &frame,
                              llvm::SmallVectorImpl<mlir::Value> &ops);

void collectLoopContinueOperands(LoweringContext &ctx, LoopFrame &frame,
                                 llvm::SmallVectorImpl<mlir::Value> &ops);

LoopFrame *getInnermostLoop(LoweringContext &ctx);

struct BreakTarget {
  ControlEntryKind kind = ControlEntryKind::Loop;
  LoopFrame *loop = nullptr;
  SwitchFrame *switchFrame = nullptr;
  explicit operator bool() const { return loop || switchFrame; }
};

BreakTarget getInnermostBreakTarget(LoweringContext &ctx);

class SwitchScopeGuard {
public:
  SwitchScopeGuard(LoweringContext &ctx, SwitchFrame frame);
  ~SwitchScopeGuard();

  SwitchFrame &frame();
  bool isValid() const { return valid; }

private:
  LoweringContext *ctx = nullptr;
  bool valid = false;
};

class LoopScopeState : public LoopScopeProvider {
public:
  LoopScopeState(LoweringContext &parentCtx,
                 llvm::ArrayRef<const clang::ValueDecl *> carried,
                 bool hasFirstIterFlag, mlir::Value firstIterInit,
                 mlir::Location loc);
  ~LoopScopeState() override;

  bool isValid() const;
  LoweringContext &prepareContext();
  LoweringContext &bodyContext();

  bool hasFirstIterFlag() const;
  unsigned getFirstIterIndex() const;
  mlir::Value getPrepareFirstIterArg() const;
  mlir::Value getBodyFirstIterArg() const;
  void setCurrentFirstIterValue(mlir::Value value);

  bool close();

  void collectBreakOperands(LoweringContext &ctx,
                            llvm::SmallVectorImpl<mlir::Value> &ops) override;
  void collectContinueOperands(
      LoweringContext &ctx, llvm::SmallVectorImpl<mlir::Value> &ops) override;

private:
  void setupPrepareContext();
  void setupBodyContext();
  void copySharedState(LoweringContext &childCtx);
  void setBlockArguments(LoweringContext &childCtx, mlir::Block *block);
  void cleanup();

  LoweringContext &parent;
  llvm::SmallVector<const clang::ValueDecl *, 8> carriedVars;
  mlir::Location loc;
  LoopSkeleton skeleton{};
  bool active = false;

  std::optional<mlir::OpBuilder> prepareBuilder;
  std::unique_ptr<LoweringContext> prepareCtx;

  std::optional<mlir::OpBuilder> bodyBuilder;
  std::unique_ptr<LoweringContext> bodyCtx;
};

} // namespace simt_hlsl_import
