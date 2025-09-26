#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "mlir/IR/Location.h"
#include "simt-hlsl-import/LoopScopeSupport.h"
#include "llvm/ADT/ArrayRef.h"

namespace clang {
class Stmt;
class ValueDecl;
}

namespace simt_hlsl_import {

struct LoweringContext;

struct AnalysisValue {
  AnalysisValue() = default;
  AnalysisValue(mlir::Value v) { setValue(v); }

  static AnalysisValue invalid() { return AnalysisValue(); }

  static AnalysisValue fromValue(mlir::Value v) {
    AnalysisValue result;
    result.setValue(v);
    return result;
  }

  static AnalysisValue fromSym(SymValue sym) {
    AnalysisValue result;
    result.setSym(std::move(sym));
    return result;
  }

  void setValue(mlir::Value v) {
    value = v;
    hasValue = (v != nullptr);
  }

  bool hasMlirValue() const { return hasValue; }
  mlir::Value getValue() const { return value; }
  mlir::Value getValueOrNull() const { return value; }

  bool isValid() const { return hasValue || hasSym || isConstant; }

  mlir::Type getType() const {
    return value ? value.getType() : mlir::Type();
  }

  explicit operator bool() const { return isValid(); }
  operator mlir::Value() const { return value; }

  void clearValue() {
    value = nullptr;
    hasValue = false;
  }

  void setSym(SymValue sym) {
    symValue = std::move(sym);
    hasSym = true;
  }

  const SymValue *getSym() const { return hasSym ? &symValue : nullptr; }
  bool hasSymValue() const { return hasSym; }

  void setConstantInt(int64_t v) {
    isConstant = true;
    intConstant = v;
    floatConstant.reset();
  }

  void setConstantFloat(double v) {
    isConstant = true;
    floatConstant = v;
    intConstant.reset();
  }

  std::optional<int64_t> getConstantInt() const { return intConstant; }
  std::optional<double> getConstantFloat() const { return floatConstant; }

private:
  mlir::Value value;
  SymValue symValue;
  bool hasSym = false;
  bool hasValue = false;
  bool isConstant = false;
  std::optional<int64_t> intConstant;
  std::optional<double> floatConstant;
};

struct SourceLoc {
  const clang::Stmt *clangNode = nullptr;
  mlir::Location mlirLoc;
};

enum class ValueClass { ScalarInt, ScalarFloat, Vector, Pointer, Unknown };

enum class ArithOp { Add, Sub, Mul, Div, Rem, Neg, BitAnd, BitOr, BitXor };

enum class CmpOp { EQ, NE, LT, LE, GT, GE };

enum class LogicalOp { And, Or };

enum class BarrierKind { Workgroup, Device, All };

enum class BufferAtomicOp {
  Add,
  Exchange,
  CompareExchange,
  Min,
  Max,
  And,
  Or,
  Xor,
};

enum class WaveIntrinsic {
  ActiveAllTrue,
  ActiveAnyTrue,
  ActiveCountBits,
  GetLaneIndex,
};

template <typename Self, typename ValueT> struct LoweringAlgebra {
  using Value = ValueT;

  Value emitConstantInt(int64_t value, const char *tag, SourceLoc loc) {
    return self().emitConstantInt(value, tag, loc);
  }

  Value emitConstantFloat(double value, const char *tag, SourceLoc loc) {
    return self().emitConstantFloat(value, tag, loc);
  }

  Value emitArithmetic(ArithOp op, Value lhs, Value rhs, SourceLoc loc) {
    return self().emitArithmetic(op, lhs, rhs, loc);
  }

  Value emitCompare(CmpOp op, Value lhs, Value rhs, SourceLoc loc) {
    return self().emitCompare(op, lhs, rhs, loc);
  }

  Value emitSelect(Value cond, Value trueV, Value falseV, SourceLoc loc) {
    return self().emitSelect(cond, trueV, falseV, loc);
  }

  template <typename... Args>
  decltype(auto) beginIf(Args &&... args) {
    return self().beginIf(std::forward<Args>(args)...);
  }

  template <typename... Args>
  decltype(auto) beginLoop(Args &&... args) {
    return self().beginLoop(std::forward<Args>(args)...);
  }

  template <typename ThenMake, typename ElseMake>
  Value emitConditional(Value cond, ThenMake &&thenBuilder, ElseMake &&elseBuilder,
                        SourceLoc loc) {
    return self().emitConditional(cond, std::forward<ThenMake>(thenBuilder),
                                  std::forward<ElseMake>(elseBuilder), loc);
  }

  template <typename RHSMake>
  Value emitShortCircuit(LogicalOp op, Value lhs, RHSMake &&rhsBuilder,
                         SourceLoc loc) {
    return self().emitShortCircuit(op, lhs,
                                   std::forward<RHSMake>(rhsBuilder), loc);
  }

  Value emitBufferLoad(Value resourceHandle, Value index,
                       const clang::ValueDecl *resourceDecl, SourceLoc loc) {
    return self().emitBufferLoad(resourceHandle, index, resourceDecl, loc);
  }

  void emitBufferStore(Value resourceHandle, Value index, Value storedValue,
                       const clang::ValueDecl *resourceDecl, SourceLoc loc) {
    self().emitBufferStore(resourceHandle, index, storedValue, resourceDecl,
                           loc);
  }

  Value emitAtomic(BufferAtomicOp op, Value resourceHandle, Value index,
                   Value value, Value compare,
                   const clang::ValueDecl *resourceDecl, SourceLoc loc) {
    return self().emitAtomic(op, resourceHandle, index, value, compare,
                             resourceDecl, loc);
  }

  Value emitWaveIntrinsic(WaveIntrinsic op, llvm::ArrayRef<Value> operands,
                          mlir::Type resultType, SourceLoc loc) {
    return self().emitWaveIntrinsic(op, operands, resultType, loc);
  }

  Value lookupVariable(const clang::ValueDecl *decl) {
    return self().lookupVariable(decl);
  }

  void bindVariable(const clang::ValueDecl *decl, Value value) {
    self().bindVariable(decl, value);
  }

  void noteMutation(const clang::ValueDecl *decl) { self().noteMutation(decl); }

  void emitReturn(std::optional<Value> value, SourceLoc loc) {
    self().emitReturn(std::move(value), loc);
  }

  void emitBarrier(BarrierKind kind, SourceLoc loc) {
    self().emitBarrier(kind, loc);
  }

  void emitFence(BarrierKind kind, const char *memSpace, SourceLoc loc) {
    self().emitFence(kind, memSpace, loc);
  }

  void trace(const char *message, SourceLoc loc) { self().trace(message, loc); }

private:
  Self &self() { return *static_cast<Self *>(this); }
  const Self &self() const { return *static_cast<const Self *>(this); }
};

struct DiagSink {
  virtual ~DiagSink() = default;
  virtual void report(SourceLoc loc, const char *message) = 0;
};

} // namespace simt_hlsl_import
