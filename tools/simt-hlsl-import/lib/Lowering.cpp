#include "simt-hlsl-import/Lowering.h"

#include "simt-step/Dialect/SimtStep/SimtStepDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/Tooling.h"

#include <algorithm>
#include <functional>
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
class SourceManager;
}

namespace simt_hlsl_import {

namespace {

enum class ControlEntryKind { Loop, Switch };

struct LoopFrame {
  simt::dialect::LoopOp loop;
  llvm::SmallVector<const clang::ValueDecl *, 8> carriedVars;
  bool hasFirstIterFlag = false;
  unsigned firstIterIndex = 0;
  mlir::Value currentFirstIterValue;
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

static mlir::Type convertType(const clang::QualType &qt,
                              mlir::OpBuilder &builder) {
  clang::LangAS addressSpace = qt.getAddressSpace();
  const clang::Type *type = qt.getCanonicalType().getTypePtrOrNull();
  if (!type)
    return {};

  auto desugar = [](const clang::Type *ty) -> const clang::Type * {
    while (auto *elab = llvm::dyn_cast<clang::ElaboratedType>(ty))
      ty = elab->getNamedType().getTypePtr();
    return ty->getUnqualifiedDesugaredType();
  };
  type = desugar(type);

  auto *tmplSpec = [&]() -> const clang::ClassTemplateSpecializationDecl * {
    if (const auto *recordType = llvm::dyn_cast<clang::RecordType>(type)) {
      const auto *cxxRecord =
          llvm::dyn_cast<clang::CXXRecordDecl>(recordType->getDecl());
      if (const auto *spec =
              llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(
                  cxxRecord))
        return spec;
    }
    if (const auto *specType =
            llvm::dyn_cast<clang::TemplateSpecializationType>(type)) {
      if (auto *templDecl = specType->getTemplateName().getAsTemplateDecl())
        if (const auto *record = llvm::dyn_cast<clang::CXXRecordDecl>(
                templDecl->getTemplatedDecl())) {
          if (const auto *spec =
                  llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                      record))
            return spec;
        }
    }
    return nullptr;
  }();

  if (!tmplSpec) {
    if (const auto *recordType = llvm::dyn_cast<clang::RecordType>(type)) {
      const auto *cxxRecord =
          llvm::dyn_cast<clang::CXXRecordDecl>(recordType->getDecl());
      if (cxxRecord) {
        llvm::SmallPtrSet<const clang::CXXRecordDecl *, 8> visited;
        std::function<const clang::ClassTemplateSpecializationDecl *(
            const clang::CXXRecordDecl *)>
            findResourceBase = [&](const clang::CXXRecordDecl *record)
                -> const clang::ClassTemplateSpecializationDecl * {
          if (!record || !visited.insert(record).second)
            return nullptr;
          if (const auto *spec =
                  llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(
                      record))
            return spec;
          for (const clang::CXXBaseSpecifier &base : record->bases()) {
            const clang::Type *baseType =
                base.getType().getCanonicalType().getTypePtrOrNull();
            if (!baseType)
              continue;
            baseType = baseType->getUnqualifiedDesugaredType();
            if (const auto *baseRecord =
                    baseType->getAsCXXRecordDecl())
              if (const auto *spec = findResourceBase(baseRecord))
                return spec;
          }
          return nullptr;
        };

        tmplSpec = findResourceBase(cxxRecord);
      }
    }
  }

  if (tmplSpec) {
    auto name = tmplSpec->getName();
    if ((name == "Buffer" || name == "RWBuffer") &&
        tmplSpec->getTemplateArgs().size() >= 1) {
      const auto &arg = tmplSpec->getTemplateArgs()[0];
      if (arg.getKind() == clang::TemplateArgument::Type) {
        mlir::Type elementType = convertType(arg.getAsType(), builder);
        if (!elementType)
          return {};
        auto memorySpace = simt::dialect::MemorySpace::Global;
        return simt::dialect::ResourceType::get(builder.getContext(),
                                                memorySpace, elementType);
      }
    }
  }

  if (const auto *recordType = llvm::dyn_cast<clang::RecordType>(type)) {
    const auto *recordDecl = recordType->getDecl();
    llvm::StringRef recordName = recordDecl->getName();
    if (recordName == "ByteAddressBuffer" || recordName == "RWByteAddressBuffer") {
      mlir::Type elementType = builder.getIntegerType(32);
      auto memorySpace = simt::dialect::MemorySpace::Global;
      return simt::dialect::ResourceType::get(builder.getContext(), memorySpace,
                                              elementType);
    }

    if (const auto *cxxRecord =
            llvm::dyn_cast<clang::CXXRecordDecl>(recordDecl)) {
      for (const auto *annot :
           cxxRecord->specific_attrs<clang::AnnotateAttr>()) {
        llvm::StringRef text = annot->getAnnotation();
        if (!text.consume_front("simt.resource:"))
          continue;

        llvm::StringRef memSpaceStr;
        llvm::StringRef elementStr;
        std::tie(memSpaceStr, elementStr) = text.split(':');
        if (elementStr.empty())
          continue;

        auto memorySpace = simt::dialect::MemorySpace::Global;
        if (memSpaceStr == "Shared")
          memorySpace = simt::dialect::MemorySpace::Shared;
        else if (memSpaceStr == "Private")
          memorySpace = simt::dialect::MemorySpace::Private;
        else if (memSpaceStr == "Generic")
          memorySpace = simt::dialect::MemorySpace::Generic;
        else if (!memSpaceStr.empty() && memSpaceStr != "Global")
          continue;

        mlir::Type elementType;
        if (elementStr == "i8")
          elementType = builder.getIntegerType(8);
        else if (elementStr == "i16")
          elementType = builder.getIntegerType(16);
        else if (elementStr == "i32")
          elementType = builder.getIntegerType(32);
        else if (elementStr == "i64")
          elementType = builder.getIntegerType(64);
        else if (elementStr == "f16")
          elementType = builder.getF16Type();
        else if (elementStr == "f32")
          elementType = builder.getF32Type();
        else if (elementStr == "f64")
          elementType = builder.getF64Type();
        else
          continue;

        return simt::dialect::ResourceType::get(builder.getContext(), memorySpace,
                                                elementType);
      }
    }
  }

  if (const auto *vectorType = llvm::dyn_cast<clang::VectorType>(type)) {
    mlir::Type elementType = convertType(vectorType->getElementType(), builder);
    if (!elementType)
      return {};

    auto numElements = static_cast<int64_t>(vectorType->getNumElements());
    return mlir::VectorType::get({numElements}, elementType);
  }

  if (const auto *arrayType = llvm::dyn_cast<clang::ArrayType>(type)) {
    if (addressSpace == clang::LangAS::hlsl_groupshared) {
      mlir::Type elementType =
          convertType(arrayType->getElementType(), builder);
      if (!elementType)
        return {};
      return simt::dialect::ResourceType::get(builder.getContext(),
                                              simt::dialect::MemorySpace::Shared,
                                              elementType);
    }
  }

  if (const auto *builtin = llvm::dyn_cast<clang::BuiltinType>(type)) {
    switch (builtin->getKind()) {
    case clang::BuiltinType::Void:
      return builder.getNoneType();
    case clang::BuiltinType::Bool:
      return builder.getI1Type();
    case clang::BuiltinType::SChar:
    case clang::BuiltinType::Char_S:
    case clang::BuiltinType::Char_U:
    case clang::BuiltinType::UChar:
      return builder.getIntegerType(8);
    case clang::BuiltinType::Short:
    case clang::BuiltinType::UShort:
      return builder.getIntegerType(16);
    case clang::BuiltinType::Int:
    case clang::BuiltinType::UInt:
    case clang::BuiltinType::Long:
    case clang::BuiltinType::ULong:
      return builder.getIntegerType(32);
    case clang::BuiltinType::LongLong:
    case clang::BuiltinType::ULongLong:
      return builder.getIntegerType(64);
    case clang::BuiltinType::Half:
      return builder.getF16Type();
    case clang::BuiltinType::Float:
      return builder.getF32Type();
    case clang::BuiltinType::Double:
      return builder.getF64Type();
    default:
      break;
    }
  }

  return {};
}

static mlir::Value buildZeroValue(LoweringContext &ctx, mlir::Type type);

struct BufferAccessInfo {
  mlir::Value resource;
  mlir::Value index;
  simt::dialect::ResourceType resourceType;
  const clang::ValueDecl *decl = nullptr;
};

enum class BufferAtomicKind {
  Add,
  Exchange,
  CompareExchange,
  Min,
  Max,
  And,
  Or,
  Xor,
};

static std::optional<BufferAccessInfo>
getBufferAccessInfo(const clang::Expr *baseExpr, const clang::Expr *indexExpr,
                    LoweringContext &ctx);

static std::optional<BufferAccessInfo>
lowerBufferAccessOperands(const clang::CXXOperatorCallExpr *opCall,
                          LoweringContext &ctx);

static mlir::Value lowerAssignment(const clang::BinaryOperator *binOp,
                                   LoweringContext &ctx);

static std::optional<mlir::Value>
lowerAtomicMemberCall(const clang::CXXMemberCallExpr *call,
                      LoweringContext &ctx);

static mlir::Location getLocation(const clang::Stmt *stmt,
                                  LoweringContext &ctx);
static mlir::Value lowerExpr(const clang::Expr *expr, LoweringContext &ctx);

static std::optional<mlir::Value>
lowerAtomicCall(const clang::CallExpr *call, LoweringContext &ctx);

static std::optional<mlir::Value>
lowerWaveIntrinsicCall(const clang::CallExpr *call, LoweringContext &ctx);

static std::optional<mlir::Value>
lowerBarrierUtilityCall(const clang::CallExpr *call, LoweringContext &ctx);

static mlir::Value emitBufferAtomicOp(BufferAtomicKind kind,
                                      const BufferAccessInfo &info,
                                      mlir::Value compareValue,
                                      mlir::Value valueValue,
                                      mlir::Location loc,
                                      LoweringContext &ctx) {
  switch (kind) {
  case BufferAtomicKind::Add:
    return ctx.builder
        .create<simt::dialect::BufferAtomicAddOp>(loc, info.resource,
                                                  info.index, valueValue)
        .getOldValue();
  case BufferAtomicKind::Exchange:
    return ctx.builder
        .create<simt::dialect::BufferAtomicExchangeOp>(loc, info.resource,
                                                        info.index, valueValue)
        .getOldValue();
  case BufferAtomicKind::CompareExchange:
    return ctx.builder
        .create<simt::dialect::BufferAtomicCompareExchangeOp>(
            loc, info.resource, info.index, compareValue, valueValue)
        .getOldValue();
  case BufferAtomicKind::Min:
    return ctx.builder
        .create<simt::dialect::BufferAtomicMinOp>(loc, info.resource,
                                                  info.index, valueValue)
        .getOldValue();
  case BufferAtomicKind::Max:
    return ctx.builder
        .create<simt::dialect::BufferAtomicMaxOp>(loc, info.resource,
                                                  info.index, valueValue)
        .getOldValue();
  case BufferAtomicKind::And:
    return ctx.builder
        .create<simt::dialect::BufferAtomicAndOp>(loc, info.resource,
                                                  info.index, valueValue)
        .getOldValue();
  case BufferAtomicKind::Or:
    return ctx.builder
        .create<simt::dialect::BufferAtomicOrOp>(loc, info.resource,
                                                 info.index, valueValue)
        .getOldValue();
  case BufferAtomicKind::Xor:
    return ctx.builder
        .create<simt::dialect::BufferAtomicXorOp>(loc, info.resource,
                                                  info.index, valueValue)
        .getOldValue();
  }

  llvm_unreachable("unknown buffer atomic kind");
}

static std::optional<BufferAccessInfo>
getBufferAccessInfoFromLValue(const clang::Expr *expr,
                              LoweringContext &ctx) {
  if (!expr)
    return std::nullopt;

  expr = expr->IgnoreParenImpCasts();

  if (const auto *subscript =
          llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
    if (subscript->getOperator() == clang::OO_Subscript)
      return lowerBufferAccessOperands(subscript, ctx);
  }

  if (const auto *arraySub = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr))
    return getBufferAccessInfo(arraySub->getBase(), arraySub->getIdx(), ctx);

  return std::nullopt;
}

static std::optional<mlir::Value>
lowerAtomicCall(const clang::CallExpr *call, LoweringContext &ctx) {
  if (!call)
    return std::nullopt;

  const auto *callee = call->getDirectCallee();
  if (!callee)
    return std::nullopt;

  auto kind = llvm::StringSwitch<std::optional<BufferAtomicKind>>(callee->getName())
                  .Case("InterlockedAdd", BufferAtomicKind::Add)
                  .Case("InterlockedExchange", BufferAtomicKind::Exchange)
                  .Case("InterlockedCompareExchange",
                        BufferAtomicKind::CompareExchange)
                  .Case("InterlockedMin", BufferAtomicKind::Min)
                  .Case("InterlockedMax", BufferAtomicKind::Max)
                  .Case("InterlockedAnd", BufferAtomicKind::And)
                  .Case("InterlockedOr", BufferAtomicKind::Or)
                  .Case("InterlockedXor", BufferAtomicKind::Xor)
                  .Default(std::nullopt);
  if (!kind)
    return std::nullopt;

  unsigned numArgs = call->getNumArgs();
  unsigned valueOperandCount =
      *kind == BufferAtomicKind::CompareExchange ? 2U : 1U;
  unsigned baseArgCount = 1 + valueOperandCount;
  if (numArgs != baseArgCount && numArgs != baseArgCount + 1)
    return ctx.fail("unexpected argument count for atomic call"),
           std::optional<mlir::Value>(mlir::Value());

  const clang::Expr *destArg = call->getArg(0);
  if (!destArg)
    return ctx.fail("atomic call missing destination argument"),
           std::optional<mlir::Value>(mlir::Value());

  destArg = destArg->IgnoreParenImpCasts();
  if (const auto *outExpr =
          llvm::dyn_cast<clang::HLSLOutArgExpr>(destArg))
    destArg = outExpr->getArgLValue()->IgnoreParenImpCasts();

  auto infoOpt = getBufferAccessInfoFromLValue(destArg, ctx);
  if (!infoOpt)
    return ctx.fail("atomic destination must be a buffer element"),
           std::optional<mlir::Value>(mlir::Value());

  BufferAccessInfo info = *infoOpt;
  mlir::Location loc = getLocation(call, ctx);

  unsigned argIndex = 1;

  mlir::Value compareValue;
  mlir::Value valueValue;
  if (*kind == BufferAtomicKind::CompareExchange) {
    compareValue = lowerExpr(call->getArg(argIndex++), ctx);
    if (!compareValue)
      return mlir::Value();
    valueValue = lowerExpr(call->getArg(argIndex++), ctx);
    if (!valueValue)
      return mlir::Value();
  } else {
    valueValue = lowerExpr(call->getArg(argIndex++), ctx);
    if (!valueValue)
      return mlir::Value();
  }

  const clang::Expr *outArg = nullptr;
  if (numArgs == baseArgCount + 1)
    outArg = call->getArg(argIndex++);

  mlir::Value oldValue =
      emitBufferAtomicOp(*kind, info, compareValue, valueValue, loc, ctx);

  if (info.decl)
    ctx.mutatedVars.insert(info.decl);

  if (outArg) {
    const clang::Expr *stripped = outArg->IgnoreParenImpCasts();
    if (const auto *outExpr =
            llvm::dyn_cast<clang::HLSLOutArgExpr>(stripped))
      stripped = outExpr->getArgLValue()->IgnoreParenImpCasts();
    const clang::ValueDecl *outDecl = nullptr;
    if (const auto *declRef =
            llvm::dyn_cast<clang::DeclRefExpr>(stripped))
      outDecl = declRef->getDecl();
    if (!outDecl)
      return ctx.fail(
                 "atomic original value argument must reference a variable"),
             std::optional<mlir::Value>(mlir::Value());
    ctx.valueMap[outDecl] = oldValue;
    ctx.mutatedVars.insert(outDecl);
  }

  return mlir::Value();
}

static std::optional<mlir::Value>
lowerWaveIntrinsicCall(const clang::CallExpr *call, LoweringContext &ctx) {
  if (!call)
    return std::nullopt;

  const auto *callee = call->getDirectCallee();
  if (!callee)
    return std::nullopt;

  llvm::StringRef name = callee->getName();
  mlir::Location loc = getLocation(call, ctx);

  if (name == "WaveActiveAllTrue" || name == "WaveActiveAnyTrue") {
    if (call->getNumArgs() != 1)
      return ctx.fail("wave intrinsic expects one argument"),
             std::optional<mlir::Value>(mlir::Value());
    mlir::Value operand = lowerExpr(call->getArg(0), ctx);
    if (!operand)
      return mlir::Value();
    mlir::Type boolType = convertType(call->getType(), ctx.builder);
    if (!boolType)
      return ctx.fail("unsupported return type for wave intrinsic"),
             std::optional<mlir::Value>(mlir::Value());
    if (name == "WaveActiveAllTrue")
      return ctx.builder
          .create<simt::dialect::WaveAllOp>(loc, boolType, operand)
          .getResult();
    return ctx.builder
        .create<simt::dialect::WaveAnyOp>(loc, boolType, operand)
        .getResult();
  }

  if (name == "WaveGetLaneIndex") {
    if (call->getNumArgs() != 0)
      return ctx.fail("WaveGetLaneIndex expects no arguments"),
             std::optional<mlir::Value>(mlir::Value());
    mlir::Type expectedType = convertType(call->getType(), ctx.builder);
    if (!expectedType)
      return ctx.fail("unsupported return type for WaveGetLaneIndex"),
             std::optional<mlir::Value>(mlir::Value());
    mlir::Value laneIndex =
        ctx.builder
            .create<simt::dialect::LaneIdOp>(loc, ctx.builder.getIndexType())
            .getLane();
    if (mlir::isa<mlir::IndexType>(expectedType))
      return laneIndex;
    if (mlir::isa<mlir::IntegerType>(expectedType))
      return ctx.builder
          .create<mlir::arith::IndexCastOp>(loc, expectedType, laneIndex)
          .getResult();
    return ctx.fail("unsupported result type for WaveGetLaneIndex"),
           std::optional<mlir::Value>(mlir::Value());
  }

  return std::nullopt;
}

static std::optional<mlir::Value>
lowerBarrierUtilityCall(const clang::CallExpr *call, LoweringContext &ctx) {
  if (!call)
    return std::nullopt;

  const auto *callee = call->getDirectCallee();
  if (!callee)
    return std::nullopt;

  llvm::StringRef name = callee->getName();
  if (name != "GroupMemoryBarrierWithGroupSync")
    return std::nullopt;

  if (call->getNumArgs() != 0)
    return ctx.fail("memory barrier utilities do not take arguments"),
           std::optional<mlir::Value>(mlir::Value());

  return mlir::Value();
}

static std::optional<std::string>
buildDxilTripleForProfile(llvm::StringRef profile) {
  llvm::SmallVector<llvm::StringRef, 4> parts;
  profile.split(parts, '_', /*MaxSplit=*/3, /*KeepEmpty=*/false);
  if (parts.empty())
    return std::nullopt;

  std::string stageLower = parts[0].lower();
  llvm::StringRef environment;
  if (stageLower == "cs")
    environment = "compute";
  else if (stageLower == "ps")
    environment = "pixel";
  else if (stageLower == "vs")
    environment = "vertex";
  else if (stageLower == "gs")
    environment = "geometry";
  else if (stageLower == "ds")
    environment = "domain";
  else if (stageLower == "hs")
    environment = "hull";
  else if (stageLower == "ms")
    environment = "mesh";
  else if (stageLower == "as")
    environment = "amplification";
  else if (stageLower == "lib" || stageLower == "library")
    environment = "library";
  else
    return std::nullopt;

  llvm::StringRef major = parts.size() > 1 ? parts[1] : "6";
  llvm::StringRef minor = parts.size() > 2 ? parts[2] : "0";
  std::string version = (llvm::Twine(major) + "." + minor).str();

  return (llvm::Twine("dxil-pc-shadermodel") + version + "-" + environment)
      .str();
}

static mlir::Location getLocation(const clang::Stmt *stmt,
                                  LoweringContext &ctx) {
  if (!stmt || !ctx.sourceManager)
    return ctx.defaultLoc;

  const clang::SourceManager &sm = *ctx.sourceManager;
  clang::SourceLocation loc = stmt->getBeginLoc();
  if (loc.isInvalid())
    loc = stmt->getEndLoc();
  if (loc.isInvalid())
    return ctx.defaultLoc;

  loc = sm.getExpansionLoc(loc);
  clang::PresumedLoc presumed = sm.getPresumedLoc(loc);
  if (!presumed.isValid())
    return ctx.defaultLoc;

  mlir::MLIRContext *mlirCtx = ctx.builder.getContext();
  mlir::StringAttr fileAttr =
      mlir::StringAttr::get(mlirCtx, presumed.getFilename());
  return mlir::FileLineColLoc::get(fileAttr, presumed.getLine(),
                                   presumed.getColumn());
}

static mlir::Value lowerExpr(const clang::Expr *expr, LoweringContext &ctx) {
  if (!expr)
    return {};

  mlir::Type type = convertType(expr->getType(), ctx.builder);
  if (!type)
    return ctx.fail("unsupported expression type"), mlir::Value();

  mlir::Location loc = getLocation(expr, ctx);

  if (const auto *intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
    if (!mlir::isa<mlir::IntegerType>(type))
      return ctx.fail("integer literal expects integer type"), mlir::Value();
    auto attr = ctx.builder.getIntegerAttr(mlir::cast<mlir::IntegerType>(type),
                                           intLit->getValue());
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }

  if (const auto *paren = llvm::dyn_cast<clang::ParenExpr>(expr))
    return lowerExpr(paren->getSubExpr(), ctx);

  if (const auto *implicitCast = llvm::dyn_cast<clang::ImplicitCastExpr>(expr))
    return lowerExpr(implicitCast->getSubExpr(), ctx);

  if (const auto *constExpr = llvm::dyn_cast<clang::ConstantExpr>(expr))
    return lowerExpr(constExpr->getSubExpr(), ctx);

  if (const auto *floatLit = llvm::dyn_cast<clang::FloatingLiteral>(expr)) {
    if (!mlir::isa<mlir::FloatType>(type))
      return ctx.fail("floating literal expects floating type"), mlir::Value();
    auto attr = ctx.builder.getFloatAttr(mlir::cast<mlir::FloatType>(type),
                                         floatLit->getValue());
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }

  if (const auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
    const clang::ValueDecl *vd = declRef->getDecl();
    auto it = ctx.valueMap.find(vd);
    if (it != ctx.valueMap.end())
      return it->second;
    return ctx.fail("reference to unknown value"), mlir::Value();
  }

  if (const auto *opCall = llvm::dyn_cast<clang::CXXOperatorCallExpr>(expr)) {
    if (opCall->getOperator() == clang::OO_Subscript) {
      auto infoOpt = lowerBufferAccessOperands(opCall, ctx);
      if (!infoOpt)
        return {};
      const auto &info = *infoOpt;
      return ctx.builder
          .create<simt::dialect::BufferLoadOp>(loc, info.resource, info.index)
          .getResult();
    }
  }

  if (const auto *memberCall = llvm::dyn_cast<clang::CXXMemberCallExpr>(expr)) {
    if (auto lowered = lowerAtomicMemberCall(memberCall, ctx))
      return *lowered;
  }

  if (const auto *callExpr = llvm::dyn_cast<clang::CallExpr>(expr)) {
    if (auto lowered = lowerAtomicCall(callExpr, ctx))
      return *lowered;
    if (auto lowered = lowerWaveIntrinsicCall(callExpr, ctx))
      return *lowered;
    if (auto lowered = lowerBarrierUtilityCall(callExpr, ctx))
      return *lowered;
  }

  if (const auto *arraySub = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
    auto infoOpt = getBufferAccessInfo(arraySub->getBase(), arraySub->getIdx(),
                                       ctx);
    if (!infoOpt)
      return {};
    const auto &info = *infoOpt;
    return ctx.builder
        .create<simt::dialect::BufferLoadOp>(loc, info.resource, info.index)
        .getResult();
  }

  if (const auto *vecElem = llvm::dyn_cast<clang::ExtVectorElementExpr>(expr)) {
    if (vecElem->isArrow())
      return ctx.fail("pointer-based vector swizzles are unsupported"),
             mlir::Value();

    mlir::Value base = lowerExpr(vecElem->getBase(), ctx);
    if (!base)
      return {};

    auto baseVecType = mlir::dyn_cast<mlir::VectorType>(base.getType());
    if (!baseVecType || baseVecType.getRank() != 1)
      return ctx.fail("vector element access requires 1-D vector operand"),
             mlir::Value();

    llvm::SmallVector<uint32_t, 4> elementIndices32;
    vecElem->getEncodedElementAccess(elementIndices32);
    llvm::SmallVector<int64_t, 4> elements;
    elements.reserve(elementIndices32.size());
    for (uint32_t idx : elementIndices32) {
      if (idx >= baseVecType.getShape()[0])
        return ctx.fail("vector element index out of range"), mlir::Value();
      elements.push_back(static_cast<int64_t>(idx));
    }
    if (elements.empty())
      return ctx.fail("vector element access with no components"),
             mlir::Value();

    if (elements.size() == 1) {
      llvm::SmallVector<int64_t, 1> position = {elements[0]};
      return ctx.builder.create<mlir::vector::ExtractOp>(loc, base, position)
          .getResult();
    }

    mlir::Type elementType = baseVecType.getElementType();
    auto resultType = mlir::VectorType::get(
        {static_cast<int64_t>(elements.size())}, elementType);
    mlir::Value result = buildZeroValue(ctx, resultType);
    if (!result)
      return {};

    for (auto [outIdx, elementIndex] : llvm::enumerate(elements)) {
      llvm::SmallVector<int64_t, 1> extractPos = {elementIndex};
      mlir::Value component =
          ctx.builder.create<mlir::vector::ExtractOp>(loc, base, extractPos)
              .getResult();
      llvm::SmallVector<int64_t, 1> insertPos = {static_cast<int64_t>(outIdx)};
      result =
          ctx.builder
              .create<mlir::vector::InsertOp>(loc, component, result, insertPos)
              .getResult();
    }

    return result;
  }

  if (const auto *unOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
    mlir::Value operand = lowerExpr(unOp->getSubExpr(), ctx);
    if (!operand)
      return {};

    auto makeIntegerConstant = [&](int64_t value,
                                   mlir::IntegerType type) -> mlir::Value {
      return ctx.builder
          .create<mlir::arith::ConstantIntOp>(loc, value, type.getWidth())
          .getResult();
    };
    auto makeFloatConstant = [&](double value,
                                 mlir::FloatType type) -> mlir::Value {
      auto attr = ctx.builder.getFloatAttr(type, value);
      return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr).getResult();
    };

    auto getMutableDeclRef =
        [&](const clang::Expr *expr) -> const clang::ValueDecl * {
      const clang::Expr *stripped = expr->IgnoreParenImpCasts();
      if (const auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(stripped))
        return ref->getDecl();
      return nullptr;
    };

    switch (unOp->getOpcode()) {
    case clang::UnaryOperatorKind::UO_Plus:
      return operand;
    case clang::UnaryOperatorKind::UO_Minus: {
      if (auto floatType = mlir::dyn_cast<mlir::FloatType>(operand.getType()))
        return ctx.builder.create<mlir::arith::NegFOp>(loc, operand)
            .getResult();
      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(operand.getType())) {
        mlir::Value zero = makeIntegerConstant(0, intType);
        return ctx.builder.create<mlir::arith::SubIOp>(loc, zero, operand)
            .getResult();
      }
      return ctx.fail("unary minus requires numeric operand"), mlir::Value();
    }
    case clang::UnaryOperatorKind::UO_LNot: {
      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(operand.getType())) {
        mlir::Value zero = makeIntegerConstant(0, intType);
        return ctx.builder
            .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq,
                                         operand, zero)
            .getResult();
      }
      if (auto floatType = mlir::dyn_cast<mlir::FloatType>(operand.getType())) {
        mlir::Value zero = makeFloatConstant(0.0, floatType);
        return ctx.builder
            .create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ,
                                         operand, zero)
            .getResult();
      }
      return ctx.fail("logical not requires scalar operand"), mlir::Value();
    }
    case clang::UnaryOperatorKind::UO_Not: {
      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(operand.getType())) {
        mlir::Value allOnes = makeIntegerConstant(-1, intType);
        return ctx.builder.create<mlir::arith::XOrIOp>(loc, operand, allOnes)
            .getResult();
      }
      return ctx.fail("bitwise not requires integer operand"), mlir::Value();
    }
    case clang::UnaryOperatorKind::UO_PreInc:
    case clang::UnaryOperatorKind::UO_PreDec:
    case clang::UnaryOperatorKind::UO_PostInc:
    case clang::UnaryOperatorKind::UO_PostDec: {
      const clang::ValueDecl *target = getMutableDeclRef(unOp->getSubExpr());
      if (!target)
        return ctx.fail("increment/decrement requires simple variable"),
               mlir::Value();

      auto it = ctx.valueMap.find(target);
      if (it == ctx.valueMap.end())
        return ctx.fail("reference to unknown value"), mlir::Value();

      mlir::Value original = operand;
      mlir::Value updated;
      bool isIncrement =
          unOp->getOpcode() == clang::UnaryOperatorKind::UO_PreInc ||
          unOp->getOpcode() == clang::UnaryOperatorKind::UO_PostInc;

      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(operand.getType())) {
        mlir::Value one = makeIntegerConstant(1, intType);
        updated =
            isIncrement
                ? ctx.builder.create<mlir::arith::AddIOp>(loc, operand, one)
                      .getResult()
                : ctx.builder.create<mlir::arith::SubIOp>(loc, operand, one)
                      .getResult();
      } else if (auto floatType =
                     mlir::dyn_cast<mlir::FloatType>(operand.getType())) {
        mlir::Value one = makeFloatConstant(1.0, floatType);
        updated =
            isIncrement
                ? ctx.builder.create<mlir::arith::AddFOp>(loc, operand, one)
                      .getResult()
                : ctx.builder.create<mlir::arith::SubFOp>(loc, operand, one)
                      .getResult();
      } else {
        return ctx.fail("increment/decrement requires numeric operand"),
               mlir::Value();
      }

      ctx.valueMap[target] = updated;
      ctx.mutatedVars.insert(target);

      bool isPost = unOp->getOpcode() == clang::UnaryOperatorKind::UO_PostInc ||
                    unOp->getOpcode() == clang::UnaryOperatorKind::UO_PostDec;
      return isPost ? original : updated;
    }
    default:
      return ctx.fail("unsupported unary operator"), mlir::Value();
    }
  }

  if (const auto *condOp = llvm::dyn_cast<clang::ConditionalOperator>(expr)) {
    mlir::Value condValue = lowerExpr(condOp->getCond(), ctx);
    if (!condValue)
      return {};
    if (condValue.getType() != ctx.builder.getI1Type())
      return ctx.fail("conditional operator requires boolean condition"),
             mlir::Value();

    mlir::Region trueRegion;
    trueRegion.emplaceBlock();
    mlir::OpBuilder trueBuilder(ctx.builder.getContext());
    trueBuilder.setInsertionPointToEnd(&trueRegion.front());
    LoweringContext trueCtx(trueBuilder, loc, ctx.returnType, ctx.errorMessage,
                            ctx.sourceManager);
    trueCtx.valueMap = ctx.valueMap;
    trueCtx.loopStack = ctx.loopStack;
    trueCtx.switchStack = ctx.switchStack;
    trueCtx.controlStack = ctx.controlStack;

    mlir::Value trueValue = lowerExpr(condOp->getTrueExpr(), trueCtx);
    if (!trueValue)
      return {};
    if (trueValue.getType() != type)
      return ctx.fail("conditional operator branch type mismatch"),
             mlir::Value();

    mlir::Region falseRegion;
    falseRegion.emplaceBlock();
    mlir::OpBuilder falseBuilder(ctx.builder.getContext());
    falseBuilder.setInsertionPointToEnd(&falseRegion.front());
    LoweringContext falseCtx(falseBuilder, loc, ctx.returnType,
                             ctx.errorMessage, ctx.sourceManager);
    falseCtx.valueMap = ctx.valueMap;
    falseCtx.loopStack = ctx.loopStack;
    falseCtx.switchStack = ctx.switchStack;
    falseCtx.controlStack = ctx.controlStack;

    mlir::Value falseValue = lowerExpr(condOp->getFalseExpr(), falseCtx);
    if (!falseValue)
      return {};
    if (falseValue.getType() != type)
      return ctx.fail("conditional operator branch type mismatch"),
             mlir::Value();

    llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutatedSet;
    mutatedSet.insert(trueCtx.mutatedVars.begin(), trueCtx.mutatedVars.end());
    mutatedSet.insert(falseCtx.mutatedVars.begin(), falseCtx.mutatedVars.end());

    llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars(
        mutatedSet.begin(), mutatedSet.end());
    llvm::sort(mutatedVars,
               [](const clang::ValueDecl *lhs, const clang::ValueDecl *rhs) {
                 return lhs < rhs;
               });

    llvm::SmallVector<mlir::Type, 8> resultTypes;
    resultTypes.push_back(type);
    for (const clang::ValueDecl *vd : mutatedVars) {
      auto lookupType =
          [&](LoweringContext &context) -> std::optional<mlir::Type> {
        auto it = context.valueMap.find(vd);
        if (it != context.valueMap.end())
          return it->second.getType();
        return std::nullopt;
      };
      std::optional<mlir::Type> branchType = lookupType(trueCtx);
      if (!branchType)
        branchType = lookupType(ctx);
      if (!branchType)
        return ctx.fail("conditional operator missing carried value"),
               mlir::Value();
      resultTypes.push_back(*branchType);
    }

    auto ifOp =
        ctx.builder.create<simt::dialect::IfOp>(loc, resultTypes, condValue,
                                                /*withElseRegion=*/true);

    auto appendOperands =
        [&](LoweringContext &branchCtx,
            llvm::SmallVectorImpl<mlir::Value> &operands) -> bool {
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value;
        if (auto it = branchCtx.valueMap.find(vd);
            it != branchCtx.valueMap.end())
          value = it->second;
        else if (auto it = ctx.valueMap.find(vd); it != ctx.valueMap.end())
          value = it->second;
        else
          return false;
        operands.push_back(value);
      }
      return true;
    };

    auto buildYield = [&](mlir::Region &region, LoweringContext &branchCtx,
                          mlir::Value branchValue) -> bool {
      llvm::SmallVector<mlir::Value, 8> operands;
      operands.push_back(branchValue);
      if (!appendOperands(branchCtx, operands))
        return false;
      mlir::OpBuilder yieldBuilder(ctx.builder.getContext());
      yieldBuilder.setInsertionPointToEnd(&region.front());
      yieldBuilder.create<simt::dialect::YieldOp>(loc, operands);
      return true;
    };

    if (!buildYield(trueRegion, trueCtx, trueValue) ||
        !buildYield(falseRegion, falseCtx, falseValue))
      return ctx.fail("conditional operator missing carried value"),
             mlir::Value();

    auto replaceRegionBody = [](mlir::Region &dest, mlir::Region &src) {
      if (!dest.empty())
        dest.front().erase();
      dest.takeBody(src);
      if (dest.empty())
        dest.emplaceBlock();
    };

    replaceRegionBody(ifOp.getThenRegion(), trueRegion);
    replaceRegionBody(ifOp.getElseRegion(), falseRegion);

    unsigned resultIndex = 1;
    for (const clang::ValueDecl *vd : mutatedVars) {
      ctx.valueMap[vd] = ifOp.getResult(resultIndex++);
      ctx.mutatedVars.insert(vd);
    }

    return ifOp.getResult(0);
  }

  if (const auto *binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    if (binOp->getOpcode() == clang::BinaryOperatorKind::BO_Assign)
      return lowerAssignment(binOp, ctx);

    mlir::Value lhs = lowerExpr(binOp->getLHS(), ctx);
    if (ctx.failed || !lhs)
      return {};

    mlir::Value rhsStorage;
    bool rhsEvaluated = false;
    auto getRHS = [&]() -> mlir::Value {
      if (!rhsEvaluated) {
        rhsStorage = lowerExpr(binOp->getRHS(), ctx);
        rhsEvaluated = true;
      }
      return rhsStorage;
    };

    switch (binOp->getOpcode()) {
    case clang::BinaryOperatorKind::BO_EQ:
    case clang::BinaryOperatorKind::BO_NE:
    case clang::BinaryOperatorKind::BO_LT:
    case clang::BinaryOperatorKind::BO_LE:
    case clang::BinaryOperatorKind::BO_GT:
    case clang::BinaryOperatorKind::BO_GE: {
      mlir::Value rhs = getRHS();
      if (ctx.failed || !rhs)
        return {};
      if (mlir::isa<mlir::IntegerType>(lhs.getType()) ||
          mlir::isa<mlir::IndexType>(lhs.getType())) {
        mlir::arith::CmpIPredicate predicate;
        switch (binOp->getOpcode()) {
        case clang::BinaryOperatorKind::BO_EQ:
          predicate = mlir::arith::CmpIPredicate::eq;
          break;
        case clang::BinaryOperatorKind::BO_NE:
          predicate = mlir::arith::CmpIPredicate::ne;
          break;
        case clang::BinaryOperatorKind::BO_LT:
          predicate = mlir::arith::CmpIPredicate::slt;
          break;
        case clang::BinaryOperatorKind::BO_LE:
          predicate = mlir::arith::CmpIPredicate::sle;
          break;
        case clang::BinaryOperatorKind::BO_GT:
          predicate = mlir::arith::CmpIPredicate::sgt;
          break;
        case clang::BinaryOperatorKind::BO_GE:
          predicate = mlir::arith::CmpIPredicate::sge;
          break;
        default:
          llvm_unreachable("unsupported integer comparison");
        }
        return ctx.builder.create<mlir::arith::CmpIOp>(loc, predicate, lhs,
                                                       rhs);
      }
      if (mlir::isa<mlir::FloatType>(lhs.getType())) {
        mlir::arith::CmpFPredicate predicate;
        switch (binOp->getOpcode()) {
        case clang::BinaryOperatorKind::BO_EQ:
          predicate = mlir::arith::CmpFPredicate::OEQ;
          break;
        case clang::BinaryOperatorKind::BO_NE:
          predicate = mlir::arith::CmpFPredicate::UNE;
          break;
        case clang::BinaryOperatorKind::BO_LT:
          predicate = mlir::arith::CmpFPredicate::OLT;
          break;
        case clang::BinaryOperatorKind::BO_LE:
          predicate = mlir::arith::CmpFPredicate::OLE;
          break;
        case clang::BinaryOperatorKind::BO_GT:
          predicate = mlir::arith::CmpFPredicate::OGT;
          break;
        case clang::BinaryOperatorKind::BO_GE:
          predicate = mlir::arith::CmpFPredicate::OGE;
          break;
        default:
          llvm_unreachable("unsupported float comparison");
        }
        return ctx.builder.create<mlir::arith::CmpFOp>(loc, predicate, lhs,
                                                       rhs);
      }
      return ctx.fail("unsupported comparison operands"), mlir::Value();
    }
    case clang::BinaryOperatorKind::BO_LAnd: {
      if (lhs.getType() != ctx.builder.getI1Type())
        return ctx.fail("logical and requires boolean operands"), mlir::Value();

      mlir::Region thenRegion;
      thenRegion.emplaceBlock();
      mlir::OpBuilder thenBuilder(ctx.builder.getContext());
      thenBuilder.setInsertionPointToEnd(&thenRegion.front());
      LoweringContext thenCtx(thenBuilder, loc, ctx.returnType,
                              ctx.errorMessage, ctx.sourceManager);
      thenCtx.valueMap = ctx.valueMap;
      thenCtx.loopStack = ctx.loopStack;
      thenCtx.switchStack = ctx.switchStack;
      thenCtx.controlStack = ctx.controlStack;

      mlir::Value rhsVal = lowerExpr(binOp->getRHS(), thenCtx);
      if (!rhsVal)
        return {};
      if (rhsVal.getType() != ctx.builder.getI1Type())
        return ctx.fail("logical and requires boolean operands"), mlir::Value();

      llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars(
          thenCtx.mutatedVars.begin(), thenCtx.mutatedVars.end());
      llvm::sort(mutatedVars, [](const clang::ValueDecl *lhsDecl,
                                 const clang::ValueDecl *rhsDecl) {
        return lhsDecl < rhsDecl;
      });

      llvm::SmallVector<mlir::Type, 8> resultTypes;
      resultTypes.push_back(ctx.builder.getI1Type());
      for (const clang::ValueDecl *vd : mutatedVars) {
        auto it = thenCtx.valueMap.find(vd);
        if (it == thenCtx.valueMap.end())
          it = ctx.valueMap.find(vd);
        if (it == ctx.valueMap.end())
          return ctx.fail("logical and missing carried value"), mlir::Value();
        resultTypes.push_back(it->second.getType());
      }

      auto ifOp = ctx.builder.create<simt::dialect::IfOp>(
          loc, resultTypes, lhs, /*withElseRegion=*/true);

      auto replaceRegionBody = [](mlir::Region &dest, mlir::Region &src) {
        if (!dest.empty())
          dest.front().erase();
        dest.takeBody(src);
        if (dest.empty())
          dest.emplaceBlock();
      };
      replaceRegionBody(ifOp.getThenRegion(), thenRegion);

      auto lookupValue = [&](LoweringContext &valueCtx,
                             const clang::ValueDecl *vd) -> mlir::Value {
        auto it = valueCtx.valueMap.find(vd);
        if (it != valueCtx.valueMap.end())
          return it->second;
        return {};
      };

      auto ensureYield = [&](mlir::Region &region,
                             llvm::ArrayRef<mlir::Value> operands) -> bool {
        auto &block = region.front();
        if (!block.empty() &&
            block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
          if (auto yield =
                  llvm::dyn_cast<simt::dialect::YieldOp>(&block.back())) {
            yield.getOperation()->setOperands(operands);
            return true;
          }
          ctx.fail("unexpected terminator while lowering logical and");
          return false;
        }
        mlir::OpBuilder::atBlockEnd(&block).create<simt::dialect::YieldOp>(
            loc, operands);
        return true;
      };

      llvm::SmallVector<mlir::Value, 8> thenOperands;
      thenOperands.push_back(rhsVal);
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = lookupValue(thenCtx, vd);
        if (!value)
          value = lookupValue(ctx, vd);
        if (!value)
          return ctx.fail("logical and missing carried value"), mlir::Value();
        thenOperands.push_back(value);
      }
      if (!ensureYield(ifOp.getThenRegion(), thenOperands))
        return {};

      auto &elseRegion = ifOp.getElseRegion();
      if (!elseRegion.empty())
        elseRegion.front().erase();
      elseRegion.emplaceBlock();
      auto elseBuilder = mlir::OpBuilder::atBlockEnd(&elseRegion.front());
      auto falseConst =
          elseBuilder.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
      llvm::SmallVector<mlir::Value, 8> elseOperands;
      elseOperands.push_back(falseConst);
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = lookupValue(ctx, vd);
        if (!value)
          return ctx.fail("logical and missing carried value"), mlir::Value();
        elseOperands.push_back(value);
      }
      if (!ensureYield(elseRegion, elseOperands))
        return {};

      unsigned resultIndex = 1;
      for (const clang::ValueDecl *vd : mutatedVars) {
        ctx.valueMap[vd] = ifOp.getResult(resultIndex++);
        ctx.mutatedVars.insert(vd);
      }

      return ifOp.getResult(0);
    }
    case clang::BinaryOperatorKind::BO_LOr: {
      if (lhs.getType() != ctx.builder.getI1Type())
        return ctx.fail("logical or requires boolean operands"), mlir::Value();

      mlir::Region elseRegion;
      elseRegion.emplaceBlock();
      mlir::OpBuilder elseBuilder(ctx.builder.getContext());
      elseBuilder.setInsertionPointToEnd(&elseRegion.front());
      LoweringContext elseCtx(elseBuilder, loc, ctx.returnType,
                              ctx.errorMessage, ctx.sourceManager);
      elseCtx.valueMap = ctx.valueMap;
      elseCtx.loopStack = ctx.loopStack;
      elseCtx.switchStack = ctx.switchStack;
      elseCtx.controlStack = ctx.controlStack;

      mlir::Value rhsVal = lowerExpr(binOp->getRHS(), elseCtx);
      if (!rhsVal)
        return {};
      if (rhsVal.getType() != ctx.builder.getI1Type())
        return ctx.fail("logical or requires boolean operands"), mlir::Value();

      llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars(
          elseCtx.mutatedVars.begin(), elseCtx.mutatedVars.end());
      llvm::sort(mutatedVars, [](const clang::ValueDecl *lhsDecl,
                                 const clang::ValueDecl *rhsDecl) {
        return lhsDecl < rhsDecl;
      });

      llvm::SmallVector<mlir::Type, 8> resultTypes;
      resultTypes.push_back(ctx.builder.getI1Type());
      for (const clang::ValueDecl *vd : mutatedVars) {
        auto it = elseCtx.valueMap.find(vd);
        if (it == elseCtx.valueMap.end())
          it = ctx.valueMap.find(vd);
        if (it == ctx.valueMap.end())
          return ctx.fail("logical or missing carried value"), mlir::Value();
        resultTypes.push_back(it->second.getType());
      }

      auto ifOp = ctx.builder.create<simt::dialect::IfOp>(
          loc, resultTypes, lhs, /*withElseRegion=*/true);

      auto replaceRegionBody = [](mlir::Region &dest, mlir::Region &src) {
        if (!dest.empty())
          dest.front().erase();
        dest.takeBody(src);
        if (dest.empty())
          dest.emplaceBlock();
      };

      auto lookupValue = [&](LoweringContext &valueCtx,
                             const clang::ValueDecl *vd) -> mlir::Value {
        auto it = valueCtx.valueMap.find(vd);
        if (it != valueCtx.valueMap.end())
          return it->second;
        return {};
      };

      auto ensureYield = [&](mlir::Region &region,
                             llvm::ArrayRef<mlir::Value> operands) -> bool {
        auto &block = region.front();
        if (!block.empty() &&
            block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
          if (auto yield =
                  llvm::dyn_cast<simt::dialect::YieldOp>(&block.back())) {
            yield.getOperation()->setOperands(operands);
            return true;
          }
          ctx.fail("unexpected terminator while lowering logical or");
          return false;
        }
        mlir::OpBuilder::atBlockEnd(&block).create<simt::dialect::YieldOp>(
            loc, operands);
        return true;
      };

      auto &thenRegion = ifOp.getThenRegion();
      if (!thenRegion.empty())
        thenRegion.front().erase();
      thenRegion.emplaceBlock();
      auto thenBuilder = mlir::OpBuilder::atBlockEnd(&thenRegion.front());
      auto trueConst =
          thenBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      llvm::SmallVector<mlir::Value, 8> thenOperands;
      thenOperands.push_back(trueConst);
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = lookupValue(ctx, vd);
        if (!value)
          return ctx.fail("logical or missing carried value"), mlir::Value();
        thenOperands.push_back(value);
      }
      if (!ensureYield(thenRegion, thenOperands))
        return {};

      replaceRegionBody(ifOp.getElseRegion(), elseRegion);
      auto &finalElseRegion = ifOp.getElseRegion();
      llvm::SmallVector<mlir::Value, 8> elseOperands;
      elseOperands.push_back(rhsVal);
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = lookupValue(elseCtx, vd);
        if (!value)
          value = lookupValue(ctx, vd);
        if (!value)
          return ctx.fail("logical or missing carried value"), mlir::Value();
        elseOperands.push_back(value);
      }
      if (!ensureYield(finalElseRegion, elseOperands))
        return {};

      unsigned resultIndex = 1;
      for (const clang::ValueDecl *vd : mutatedVars) {
        ctx.valueMap[vd] = ifOp.getResult(resultIndex++);
        ctx.mutatedVars.insert(vd);
      }

      return ifOp.getResult(0);
    }
    case clang::BinaryOperatorKind::BO_Add: {
      mlir::Value rhs = getRHS();
      if (ctx.failed || !rhs)
        return {};
      if (mlir::isa<mlir::IntegerType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
      if (mlir::isa<mlir::FloatType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::AddFOp>(loc, lhs, rhs);
      break;
    }
    case clang::BinaryOperatorKind::BO_Sub: {
      mlir::Value rhs = getRHS();
      if (ctx.failed || !rhs)
        return {};
      if (mlir::isa<mlir::IntegerType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
      if (mlir::isa<mlir::FloatType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::SubFOp>(loc, lhs, rhs);
      break;
    }
    case clang::BinaryOperatorKind::BO_Mul: {
      mlir::Value rhs = getRHS();
      if (ctx.failed || !rhs)
        return {};
      if (mlir::isa<mlir::IntegerType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
      if (mlir::isa<mlir::FloatType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::MulFOp>(loc, lhs, rhs);
      break;
    }
    case clang::BinaryOperatorKind::BO_Div: {
      mlir::Value rhs = getRHS();
      if (ctx.failed || !rhs)
        return {};
      if (mlir::isa<mlir::IntegerType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
      if (mlir::isa<mlir::FloatType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::DivFOp>(loc, lhs, rhs);
      break;
    }
    case clang::BinaryOperatorKind::BO_Rem: {
      mlir::Value rhs = getRHS();
      if (ctx.failed || !rhs)
        return {};
      if (mlir::isa<mlir::IntegerType>(lhs.getType()))
        return ctx.builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
      break;
    }
    default:
      break;
    }

    return ctx.fail("unsupported binary operator"), mlir::Value();
  }

  return ctx.fail("unsupported expression lowering"), mlir::Value();
}

static std::optional<BufferAccessInfo>
getBufferAccessInfo(const clang::Expr *baseExpr, const clang::Expr *indexExpr,
                    LoweringContext &ctx) {
  if (!baseExpr || !indexExpr)
    return std::nullopt;

  baseExpr = baseExpr->IgnoreParenImpCasts();
  indexExpr = indexExpr->IgnoreParenImpCasts();

  mlir::Value resource = lowerExpr(baseExpr, ctx);
  if (!resource)
    return std::nullopt;

  auto resourceType =
      mlir::dyn_cast<simt::dialect::ResourceType>(resource.getType());
  if (!resourceType)
    return ctx.fail("subscript base must be a buffer resource"), std::nullopt;

  mlir::Value index = lowerExpr(indexExpr, ctx);
  if (!index)
    return std::nullopt;
  if (!mlir::isa<mlir::IntegerType>(index.getType()))
    return ctx.fail("buffer subscript index must be integer"), std::nullopt;

  const clang::ValueDecl *decl = nullptr;
  if (const auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(baseExpr))
    decl = declRef->getDecl();

  return BufferAccessInfo{resource, index, resourceType, decl};
}

static std::optional<BufferAccessInfo>
lowerBufferAccessOperands(const clang::CXXOperatorCallExpr *opCall,
                          LoweringContext &ctx) {
  if (!opCall || opCall->getNumArgs() < 2)
    return std::nullopt;
  return getBufferAccessInfo(opCall->getArg(0), opCall->getArg(1), ctx);
}

static std::optional<mlir::Value>
lowerAtomicMemberCall(const clang::CXXMemberCallExpr *call,
                      LoweringContext &ctx) {
  if (!call)
    return std::nullopt;

  const auto *methodDecl = call->getMethodDecl();
  if (!methodDecl)
    return std::nullopt;

  auto kind = llvm::StringSwitch<std::optional<BufferAtomicKind>>(
                  methodDecl->getName())
                  .Case("InterlockedAdd", BufferAtomicKind::Add)
                  .Case("InterlockedExchange", BufferAtomicKind::Exchange)
                  .Case("InterlockedCompareExchange",
                        BufferAtomicKind::CompareExchange)
                  .Case("InterlockedMin", BufferAtomicKind::Min)
                  .Case("InterlockedMax", BufferAtomicKind::Max)
                  .Case("InterlockedAnd", BufferAtomicKind::And)
                  .Case("InterlockedOr", BufferAtomicKind::Or)
                  .Case("InterlockedXor", BufferAtomicKind::Xor)
                  .Default(std::nullopt);
  if (!kind)
    return std::nullopt;

  const clang::Expr *objectExpr = call->getImplicitObjectArgument();
  if (!objectExpr)
    return ctx.fail("atomic call requires an object expression"),
           std::optional<mlir::Value>(mlir::Value());

  mlir::Value resource = lowerExpr(objectExpr, ctx);
  if (!resource)
    return mlir::Value();

  auto resourceType =
      mlir::dyn_cast<simt::dialect::ResourceType>(resource.getType());
  if (!resourceType)
    return ctx.fail("atomic call requires a buffer resource"),
           std::optional<mlir::Value>(mlir::Value());

  mlir::Location loc = getLocation(call, ctx);

  unsigned numArgs = call->getNumArgs();
  unsigned valueOperandCount =
      *kind == BufferAtomicKind::CompareExchange ? 2U : 1U;
  unsigned baseArgCount = 1 + valueOperandCount;
  if (numArgs != baseArgCount && numArgs != baseArgCount + 1)
    return ctx.fail("unexpected argument count for atomic call"),
           std::optional<mlir::Value>(mlir::Value());

  unsigned argIndex = 0;
  mlir::Value indexValue = lowerExpr(call->getArg(argIndex++), ctx);
  if (!indexValue)
    return mlir::Value();

  BufferAccessInfo info{resource, indexValue, resourceType, nullptr};

  mlir::Value compareValue;
  mlir::Value valueValue;
  if (*kind == BufferAtomicKind::CompareExchange) {
    compareValue = lowerExpr(call->getArg(argIndex++), ctx);
    if (!compareValue)
      return mlir::Value();
    valueValue = lowerExpr(call->getArg(argIndex++), ctx);
    if (!valueValue)
      return mlir::Value();
  } else {
    valueValue = lowerExpr(call->getArg(argIndex++), ctx);
    if (!valueValue)
      return mlir::Value();
  }

  const clang::Expr *outArg = nullptr;
  if (numArgs == baseArgCount + 1)
    outArg = call->getArg(argIndex++);

  mlir::Value oldValue =
      emitBufferAtomicOp(*kind, info, compareValue, valueValue, loc, ctx);

  if (const auto *declRef =
          llvm::dyn_cast<clang::DeclRefExpr>(objectExpr->IgnoreParenImpCasts()))
    ctx.mutatedVars.insert(declRef->getDecl());

  if (outArg) {
    const clang::Expr *stripped = outArg->IgnoreParenImpCasts();
    if (const auto *outExpr =
            llvm::dyn_cast<clang::HLSLOutArgExpr>(stripped))
      stripped = outExpr->getArgLValue()->IgnoreParenImpCasts();
    const clang::ValueDecl *outDecl = nullptr;
    if (const auto *declRef =
            llvm::dyn_cast<clang::DeclRefExpr>(stripped))
      outDecl = declRef->getDecl();
    if (!outDecl)
      return ctx.fail(
                 "atomic original value argument must reference a variable"),
             std::optional<mlir::Value>(mlir::Value());
    ctx.valueMap[outDecl] = oldValue;
    ctx.mutatedVars.insert(outDecl);
  }

  return mlir::Value();
}

static mlir::Value lowerAssignment(const clang::BinaryOperator *binOp,
                                   LoweringContext &ctx) {
  mlir::Location loc = getLocation(binOp, ctx);
  mlir::Value rhs = lowerExpr(binOp->getRHS(), ctx);
  if (!rhs)
    return {};

  const clang::Expr *lhsExpr = binOp->getLHS()->IgnoreParenImpCasts();
  if (const auto *lhsDeclRef = llvm::dyn_cast<clang::DeclRefExpr>(lhsExpr)) {
    const clang::ValueDecl *vd = lhsDeclRef->getDecl();
    auto it = ctx.valueMap.find(vd);
    if (it != ctx.valueMap.end()) {
      it->second = rhs;
      ctx.mutatedVars.insert(vd);
      return rhs;
    }
    return ctx.fail("reference to unknown value"), mlir::Value();
  }

  if (const auto *subscript =
          llvm::dyn_cast<clang::CXXOperatorCallExpr>(lhsExpr)) {
    if (subscript->getOperator() == clang::OO_Subscript) {
      auto infoOpt = lowerBufferAccessOperands(subscript, ctx);
      if (!infoOpt)
        return {};
      const auto &info = *infoOpt;
      if (rhs.getType() != info.resourceType.getElementType())
        return ctx.fail("assignment value must match buffer element type"),
               mlir::Value();
      ctx.builder.create<simt::dialect::BufferStoreOp>(loc, info.resource,
                                                       info.index, rhs);
      if (info.decl)
        ctx.mutatedVars.insert(info.decl);
      return rhs;
    }
  }

  if (const auto *arraySub = llvm::dyn_cast<clang::ArraySubscriptExpr>(lhsExpr)) {
    auto infoOpt =
        getBufferAccessInfo(arraySub->getBase(), arraySub->getIdx(), ctx);
    if (!infoOpt)
      return {};
    const auto &info = *infoOpt;
    if (rhs.getType() != info.resourceType.getElementType())
      return ctx.fail("assignment value must match buffer element type"),
             mlir::Value();
    ctx.builder.create<simt::dialect::BufferStoreOp>(loc, info.resource,
                                                     info.index, rhs);
    if (info.decl)
      ctx.mutatedVars.insert(info.decl);
    return rhs;
  }

  return ctx.fail("unsupported assignment target"), mlir::Value();
}

static mlir::Value buildZeroValue(LoweringContext &ctx, mlir::Type type) {
  mlir::Location loc = ctx.defaultLoc;
  if (mlir::isa<mlir::IntegerType>(type)) {
    auto attr =
        ctx.builder.getIntegerAttr(mlir::cast<mlir::IntegerType>(type), 0);
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }
  if (mlir::isa<mlir::FloatType>(type)) {
    auto attr =
        ctx.builder.getFloatAttr(mlir::cast<mlir::FloatType>(type), 0.0);
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, attr);
  }
  if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(type)) {
    mlir::Type elementType = vectorType.getElementType();
    mlir::Attribute elementAttr;
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType))
      elementAttr = ctx.builder.getIntegerAttr(intType, 0);
    else if (auto floatType = mlir::dyn_cast<mlir::FloatType>(elementType))
      elementAttr = ctx.builder.getFloatAttr(floatType, 0.0);
    else
      return ctx.fail("unable to build default value for return type"),
             mlir::Value();

    auto zeroAttr = mlir::DenseElementsAttr::get(vectorType, elementAttr);
    return ctx.builder.create<mlir::arith::ConstantOp>(loc, zeroAttr);
  }
  ctx.fail("unable to build default value for return type");
  return {};
}

static bool lowerStatement(const clang::Stmt *stmt, LoweringContext &ctx);
static mlir::Value lowerExpr(const clang::Expr *expr, LoweringContext &ctx);
static bool lowerForStmt(const clang::ForStmt *stmt, LoweringContext &ctx);
static bool lowerWhileStmt(const clang::WhileStmt *stmt, LoweringContext &ctx);
static bool lowerDoStmt(const clang::DoStmt *stmt, LoweringContext &ctx);
static bool lowerSwitchStmt(const clang::SwitchStmt *stmt,
                            LoweringContext &ctx);

static void lowerCompoundStmt(const clang::CompoundStmt *compound,
                              LoweringContext &ctx) {
  for (const clang::Stmt *child : compound->body()) {
    if (ctx.emittedTerminator || ctx.failed)
      break;
    lowerStatement(child, ctx);
  }
}

static mlir::Value getLoopCarriedValue(const LoweringContext &ctx,
                                       const clang::ValueDecl *vd) {
  auto it = ctx.valueMap.find(vd);
  if (it != ctx.valueMap.end())
    return it->second;
  return {};
}

struct LoopSkeleton {
  simt::dialect::LoopOp loop;
  LoopFrame *frame = nullptr;
  mlir::Block *prepareBlock = nullptr;
  mlir::Block *bodyBlock = nullptr;
};

static bool collectLoopMutations(
    LoweringContext &ctx, const clang::Stmt *body,
    llvm::function_ref<bool(LoweringContext &)> extraWork,
    llvm::SmallVector<const clang::ValueDecl *, 8> &mutatedVars) {
  mutatedVars.clear();

  mlir::Region analysisRegion;
  analysisRegion.emplaceBlock();
  mlir::OpBuilder analysisBuilder(ctx.builder.getContext());
  analysisBuilder.setInsertionPointToStart(&analysisRegion.front());

  LoweringContext analysisCtx(analysisBuilder, ctx.defaultLoc, ctx.returnType,
                              ctx.errorMessage, ctx.sourceManager);
  analysisCtx.valueMap = ctx.valueMap;
  analysisCtx.loopStack = ctx.loopStack;
  analysisCtx.switchStack = ctx.switchStack;
  analysisCtx.controlStack = ctx.controlStack;

  if (body) {
    if (!lowerStatement(body, analysisCtx) || analysisCtx.failed)
      return false;
  }

  if (!analysisCtx.emittedTerminator) {
    if (!extraWork(analysisCtx) || analysisCtx.failed)
      return false;
  }

  llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutatedSet;
  mutatedSet.insert(analysisCtx.mutatedVars.begin(),
                    analysisCtx.mutatedVars.end());

  mutatedVars.assign(mutatedSet.begin(), mutatedSet.end());
  llvm::sort(mutatedVars,
             [](const clang::ValueDecl *lhs, const clang::ValueDecl *rhs) {
               return lhs < rhs;
             });

  return true;
}

static bool buildLoopSkeleton(
    LoweringContext &ctx, llvm::ArrayRef<const clang::ValueDecl *> mutatedVars,
    bool hasFirstIterFlag, mlir::Value firstIterInit, LoopSkeleton &out) {
  mlir::Location loc = ctx.defaultLoc;

  llvm::SmallVector<mlir::Type, 8> resultTypes;
  llvm::SmallVector<mlir::Value, 8> initValues;
  resultTypes.reserve(mutatedVars.size() + (hasFirstIterFlag ? 1 : 0));
  initValues.reserve(mutatedVars.size() + (hasFirstIterFlag ? 1 : 0));

  for (const clang::ValueDecl *vd : mutatedVars) {
    mlir::Value initial = getLoopCarriedValue(ctx, vd);
    if (!initial)
      return ctx.fail("reference to unknown loop variable");
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

  LoopFrame frame{loop, {}};
  frame.carriedVars.append(mutatedVars.begin(), mutatedVars.end());
  frame.hasFirstIterFlag = hasFirstIterFlag;
  if (hasFirstIterFlag) {
    frame.firstIterIndex = mutatedVars.size();
    frame.currentFirstIterValue = firstIterInit;
  }
  ctx.loopStack.push_back(frame);
  ctx.controlStack.push_back(
      {ControlEntryKind::Loop, ctx.loopStack.size() - 1});
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

static bool lowerStatement(const clang::Stmt *stmt, LoweringContext &ctx) {
  if (ctx.failed)
    return false;

  if (const auto *compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
    lowerCompoundStmt(compound, ctx);
    return true;
  }

  if (const auto *forStmt = llvm::dyn_cast<clang::ForStmt>(stmt))
    return lowerForStmt(forStmt, ctx);

  if (const auto *whileStmt = llvm::dyn_cast<clang::WhileStmt>(stmt))
    return lowerWhileStmt(whileStmt, ctx);

  if (const auto *doStmt = llvm::dyn_cast<clang::DoStmt>(stmt))
    return lowerDoStmt(doStmt, ctx);

  if (const auto *switchStmt = llvm::dyn_cast<clang::SwitchStmt>(stmt))
    return lowerSwitchStmt(switchStmt, ctx);

  if (llvm::isa<clang::BreakStmt>(stmt)) {
    for (auto it = ctx.controlStack.rbegin(); it != ctx.controlStack.rend();
         ++it) {
      if (it->kind == ControlEntryKind::Switch) {
        if (it->index >= ctx.switchStack.size())
          continue;
        auto &frame = ctx.switchStack[it->index];
        if (frame.analysisOnly) {
          ctx.emittedTerminator = true;
          return true;
        }

        llvm::SmallVector<mlir::Value, 8> yieldOperands;
        yieldOperands.reserve(frame.carriedVars.size() + 3);
        for (auto [index, vd] : llvm::enumerate(frame.carriedVars)) {
          mlir::Value value = ctx.valueMap.lookup(vd);
          if (!value && index < frame.initialValues.size())
            value = frame.initialValues[index];
          if (!value) {
            ctx.fail("switch break missing value for case variable");
            return false;
          }
          yieldOperands.push_back(value);
          ctx.mutatedVars.insert(vd);
        }

        auto ensureBool = [&](mlir::Value v, int constant) -> mlir::Value {
          if (v)
            return v;
          return ctx.builder.create<mlir::arith::ConstantIntOp>(ctx.defaultLoc,
                                                                constant, 1);
        };

        yieldOperands.push_back(ensureBool(frame.breakHasMatchedValue, 1));
        yieldOperands.push_back(ensureBool(frame.breakExecutingValue, 0));
        yieldOperands.push_back(ensureBool(frame.breakCompletedValue, 1));

        ctx.builder.create<simt::dialect::YieldOp>(ctx.defaultLoc,
                                                   yieldOperands);
        ctx.emittedTerminator = true;
        return true;
      }
      if (it->kind == ControlEntryKind::Loop) {
        if (it->index >= ctx.loopStack.size())
          continue;
        auto &frame = ctx.loopStack[it->index];
        llvm::SmallVector<mlir::Value, 8> operands;
        operands.reserve(frame.carriedVars.size() +
                         (frame.hasFirstIterFlag ? 1 : 0));
        for (auto [index, vd] : llvm::enumerate(frame.carriedVars)) {
          mlir::Value value = ctx.valueMap.lookup(vd);
          if (!value)
            value = frame.loop.getResult(index);
          operands.push_back(value);
        }
        if (frame.hasFirstIterFlag) {
          mlir::Value flag = frame.currentFirstIterValue;
          if (!flag && frame.firstIterIndex < frame.loop.getNumResults())
            flag = frame.loop.getResult(frame.firstIterIndex);
          operands.push_back(flag);
        }
        ctx.builder.create<simt::dialect::BreakOp>(ctx.defaultLoc, operands);
        ctx.mutatedVars.insert(frame.carriedVars.begin(),
                               frame.carriedVars.end());
        ctx.emittedTerminator = true;
        return true;
      }
    }

    return true;
  }

  if (llvm::isa<clang::ContinueStmt>(stmt)) {
    LoopFrame *loopFrame = nullptr;
    for (auto it = ctx.controlStack.rbegin(); it != ctx.controlStack.rend();
         ++it) {
      if (it->kind == ControlEntryKind::Loop) {
        if (it->index >= ctx.loopStack.size())
          continue;
        loopFrame = &ctx.loopStack[it->index];
        break;
      }
    }
    if (!loopFrame)
      return true;
    auto &frame = *loopFrame;
    llvm::SmallVector<mlir::Value, 8> operands;
    operands.reserve(frame.carriedVars.size() +
                     (frame.hasFirstIterFlag ? 1 : 0));
    for (auto [index, vd] : llvm::enumerate(frame.carriedVars)) {
      mlir::Value value = ctx.valueMap.lookup(vd);
      if (!value)
        value = frame.loop.getResult(index);
      operands.push_back(value);
    }
    if (frame.hasFirstIterFlag) {
      mlir::Value flag = frame.currentFirstIterValue;
      if (!flag && frame.firstIterIndex < frame.loop.getNumResults())
        flag = frame.loop.getResult(frame.firstIterIndex);
      operands.push_back(flag);
    }
    ctx.builder.create<simt::dialect::ContinueOp>(ctx.defaultLoc, operands);
    ctx.mutatedVars.insert(frame.carriedVars.begin(), frame.carriedVars.end());
    ctx.emittedTerminator = true;
    return true;
  }

  if (const auto *ifStmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
    mlir::Value cond = lowerExpr(ifStmt->getCond(), ctx);
    if (!cond)
      return false;

    mlir::Location loc = ctx.defaultLoc;
    bool hasElse = ifStmt->getElse() != nullptr;

    auto lowerIntoRegion = [&](const clang::Stmt *body, mlir::Region &region,
                               std::optional<mlir::OpBuilder> &builderStorage,
                               std::optional<LoweringContext> &ctxStorage,
                               LoweringContext *&outCtx) -> bool {
      region.emplaceBlock();
      builderStorage.emplace(ctx.builder.getContext());
      builderStorage->setInsertionPointToEnd(&region.front());
      ctxStorage.emplace(*builderStorage, loc, ctx.returnType, ctx.errorMessage,
                         ctx.sourceManager);
      outCtx = &*ctxStorage;
      outCtx->valueMap = ctx.valueMap;
      outCtx->mutatedVars.clear();
      outCtx->loopStack = ctx.loopStack;
      outCtx->switchStack = ctx.switchStack;
      outCtx->controlStack = ctx.controlStack;
      if (body && !lowerStatement(body, *outCtx))
        return false;
      return !outCtx->failed;
    };

    mlir::Region tmpThen;
    std::optional<mlir::OpBuilder> tmpThenBuilderStorage;
    std::optional<LoweringContext> tmpThenCtxStorage;
    LoweringContext *tmpThenCtx = nullptr;
    if (!lowerIntoRegion(ifStmt->getThen(), tmpThen, tmpThenBuilderStorage,
                         tmpThenCtxStorage, tmpThenCtx))
      return false;

    std::optional<mlir::Region> tmpElse;
    std::optional<mlir::OpBuilder> tmpElseBuilderStorage;
    std::optional<LoweringContext> tmpElseCtxStorage;
    LoweringContext *tmpElseCtx = nullptr;
    if (hasElse) {
      tmpElse.emplace();
      if (!lowerIntoRegion(ifStmt->getElse(), *tmpElse, tmpElseBuilderStorage,
                           tmpElseCtxStorage, tmpElseCtx))
        return false;
    }

    llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutatedSet;
    llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars;
    auto addMutations =
        [&](const llvm::SmallPtrSet<const clang::ValueDecl *, 8> &source) {
          for (const clang::ValueDecl *vd : source)
            if (mutatedSet.insert(vd).second)
              mutatedVars.push_back(vd);
        };
    addMutations(tmpThenCtx->mutatedVars);
    if (tmpElseCtx)
      addMutations(tmpElseCtx->mutatedVars);

    llvm::SmallVector<mlir::Type, 8> resultTypes;
    resultTypes.reserve(mutatedVars.size());
    for (const clang::ValueDecl *vd : mutatedVars)
      resultTypes.push_back(ctx.valueMap.lookup(vd).getType());

    bool needElseRegion = hasElse || !mutatedVars.empty();
    auto ifOp = ctx.builder.create<simt::dialect::IfOp>(loc, resultTypes, cond,
                                                        needElseRegion);

    auto replaceRegionBody = [](mlir::Region &dest, mlir::Region &src) {
      if (!dest.empty())
        dest.front().erase();
      dest.takeBody(src);
      if (dest.empty())
        dest.emplaceBlock();
    };

    auto &finalThen = ifOp.getThenRegion();
    replaceRegionBody(finalThen, tmpThen);

    mlir::Region *finalElse = nullptr;
    if (needElseRegion) {
      finalElse = &ifOp.getElseRegion();
      if (tmpElse)
        replaceRegionBody(*finalElse, *tmpElse);
      else {
        if (!finalElse->empty())
          finalElse->front().erase();
        finalElse->emplaceBlock();
      }
    }

    if (!mutatedVars.empty()) {
      auto materializeYield = [&](mlir::Region &region,
                                  const LoweringContext *branchCtx) {
        llvm::SmallVector<mlir::Value, 8> operands;
        operands.reserve(mutatedVars.size());
        for (const clang::ValueDecl *vd : mutatedVars) {
          if (branchCtx) {
            auto it = branchCtx->valueMap.find(vd);
            if (it != branchCtx->valueMap.end()) {
              operands.push_back(it->second);
              continue;
            }
          }
          operands.push_back(ctx.valueMap.lookup(vd));
        }
        auto &block = region.front();
        mlir::Operation *maybeTerm = block.empty() ? nullptr : &block.back();
        if (!maybeTerm || !maybeTerm->hasTrait<mlir::OpTrait::IsTerminator>())
          maybeTerm = mlir::OpBuilder::atBlockEnd(&block)
                          .create<simt::dialect::YieldOp>(loc)
                          .getOperation();
        if (auto yield = llvm::dyn_cast<simt::dialect::YieldOp>(maybeTerm))
          yield.getOperation()->setOperands(operands);
      };

      materializeYield(finalThen, tmpThenCtx);
      if (finalElse)
        materializeYield(*finalElse, tmpElseCtx);
    }

    for (auto [index, vd] : llvm::enumerate(mutatedVars)) {
      ctx.valueMap[vd] = ifOp.getResult(index);
      ctx.mutatedVars.insert(vd);
    }

    if (tmpThenCtx->emittedTerminator &&
        (!needElseRegion || (tmpElseCtx && tmpElseCtx->emittedTerminator)))
      ctx.emittedTerminator = true;

    return true;
  }

  if (const auto *ret = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
    bool expectsValue = static_cast<bool>(ctx.returnType) &&
                        !mlir::isa<mlir::NoneType>(ctx.returnType);
    if (ret->getRetValue() != nullptr) {
      mlir::Value value = lowerExpr(ret->getRetValue(), ctx);
      if (!value)
        return false;
      if (!expectsValue)
        return ctx.fail("unexpected return value in void function");
      if (value.getType() != ctx.returnType)
        return ctx.fail("return type mismatch");
      ctx.builder.create<mlir::func::ReturnOp>(getLocation(ret, ctx), value);
    } else {
      if (expectsValue)
        return ctx.fail("missing return value");
      ctx.builder.create<mlir::func::ReturnOp>(getLocation(ret, ctx));
    }
    ctx.emittedTerminator = true;
    return true;
  }

  if (const auto *declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
    for (const clang::Decl *decl : declStmt->decls()) {
      const auto *var = llvm::dyn_cast<clang::VarDecl>(decl);
      if (!var)
        continue;
      if (!convertType(var->getType(), ctx.builder))
        return ctx.fail("unsupported variable type");
      mlir::Value initValue;
      if (const clang::Expr *init = var->getInit()) {
        initValue = lowerExpr(init, ctx);
        if (!initValue)
          return false;
      }
      if (initValue)
        ctx.valueMap[var] = initValue;
    }
    return true;
  }

  if (const auto *exprStmt = llvm::dyn_cast<clang::Expr>(stmt)) {
    (void)lowerExpr(exprStmt, ctx);
    return !ctx.failed;
  }

  return ctx.fail("unsupported statement");
}

static bool lowerForStmt(const clang::ForStmt *forStmt, LoweringContext &ctx) {
  mlir::Location loc = ctx.defaultLoc;

  if (const clang::Stmt *init = forStmt->getInit()) {
    if (llvm::isa<clang::DeclStmt>(init)) {
      if (!lowerStatement(init, ctx) || ctx.failed)
        return false;
    } else if (const auto *initExpr = llvm::dyn_cast<clang::Expr>(init)) {
      (void)lowerExpr(initExpr, ctx);
      if (ctx.failed)
        return false;
    } else {
      return ctx.fail("unsupported for-loop initializer");
    }
  }

  llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars;
  const clang::Stmt *incStmt = forStmt->getInc();
  auto analyzeIncrement = [&](LoweringContext &analysisCtx) -> bool {
    if (!incStmt)
      return true;
    if (const auto *incExpr = llvm::dyn_cast<clang::Expr>(incStmt)) {
      (void)lowerExpr(incExpr, analysisCtx);
      return !analysisCtx.failed;
    }
    if (!lowerStatement(incStmt, analysisCtx) || analysisCtx.failed)
      return false;
    return true;
  };
  if (!collectLoopMutations(ctx, forStmt->getBody(), analyzeIncrement,
                            mutatedVars))
    return false;

  LoopSkeleton skeleton;
  if (!buildLoopSkeleton(ctx, mutatedVars, /*hasFirstIterFlag=*/false,
                         mlir::Value(), skeleton))
    return false;
  auto loopOp = skeleton.loop;
  auto *prepareBlock = skeleton.prepareBlock;
  auto *bodyBlock = skeleton.bodyBlock;

  // Prepare region: evaluate loop condition.
  {
    mlir::OpBuilder prepBuilder(ctx.builder.getContext());
    prepBuilder.setInsertionPointToStart(prepareBlock);
    LoweringContext prepCtx(prepBuilder, loc, ctx.returnType, ctx.errorMessage,
                            ctx.sourceManager);
    prepCtx.valueMap = ctx.valueMap;
    prepCtx.loopStack = ctx.loopStack;
    prepCtx.switchStack = ctx.switchStack;
    prepCtx.controlStack = ctx.controlStack;
    prepCtx.switchStack = ctx.switchStack;
    prepCtx.controlStack = ctx.controlStack;
    prepCtx.switchStack = ctx.switchStack;
    prepCtx.controlStack = ctx.controlStack;
    for (auto [index, vd] : llvm::enumerate(mutatedVars))
      prepCtx.valueMap[vd] = prepareBlock->getArgument(index);

    mlir::Value condValue;
    if (const clang::Expr *condExpr = forStmt->getCond()) {
      condValue = lowerExpr(condExpr, prepCtx);
      if (!condValue || prepCtx.failed)
        return false;
    } else {
      condValue = prepBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    }

    llvm::SmallVector<mlir::Value, 8> forwarded;
    forwarded.reserve(mutatedVars.size());
    for (const clang::ValueDecl *vd : mutatedVars)
      forwarded.push_back(prepCtx.valueMap.lookup(vd));

    prepBuilder.create<simt::dialect::ConditionOp>(loc, condValue, forwarded);
  }

  // Body region: execute loop body and increment.
  {
    mlir::OpBuilder bodyBuilder(ctx.builder.getContext());
    bodyBuilder.setInsertionPointToStart(bodyBlock);
    LoweringContext bodyCtx(bodyBuilder, loc, ctx.returnType, ctx.errorMessage,
                            ctx.sourceManager);
    bodyCtx.valueMap = ctx.valueMap;
    bodyCtx.loopStack = ctx.loopStack;
    bodyCtx.switchStack = ctx.switchStack;
    bodyCtx.controlStack = ctx.controlStack;
    bodyCtx.switchStack = ctx.switchStack;
    bodyCtx.controlStack = ctx.controlStack;
    for (auto [index, vd] : llvm::enumerate(mutatedVars))
      bodyCtx.valueMap[vd] = bodyBlock->getArgument(index);

    if (const clang::Stmt *body = forStmt->getBody()) {
      if (!lowerStatement(body, bodyCtx) || bodyCtx.failed)
        return false;
    }

    if (!bodyCtx.emittedTerminator) {
      if (const auto *incExpr =
              llvm::dyn_cast_or_null<clang::Expr>(forStmt->getInc())) {
        (void)lowerExpr(incExpr, bodyCtx);
        if (bodyCtx.failed)
          return false;
      } else if (const clang::Stmt *incStmt = forStmt->getInc()) {
        if (!lowerStatement(incStmt, bodyCtx) || bodyCtx.failed)
          return false;
      }
    }

    if (!bodyCtx.emittedTerminator) {
      llvm::SmallVector<mlir::Value, 8> yieldOperands;
      yieldOperands.reserve(mutatedVars.size());
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = bodyCtx.valueMap.lookup(vd);
        if (!value)
          value = ctx.valueMap.lookup(vd);
        yieldOperands.push_back(value);
      }
      bodyBuilder.create<simt::dialect::YieldOp>(loc, yieldOperands);
    }
  }

  for (auto [index, vd] : llvm::enumerate(mutatedVars)) {
    ctx.valueMap[vd] = loopOp.getResult(index);
    ctx.mutatedVars.insert(vd);
  }

  ctx.loopStack.pop_back();
  if (!ctx.controlStack.empty() &&
      ctx.controlStack.back().kind == ControlEntryKind::Loop)
    ctx.controlStack.pop_back();

  return true;
}

static bool lowerWhileStmt(const clang::WhileStmt *whileStmt,
                           LoweringContext &ctx) {
  mlir::Location loc = ctx.defaultLoc;

  const clang::Expr *condExpr = whileStmt->getCond();
  llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars;
  auto analyzeCond = [&](LoweringContext &analysisCtx) -> bool {
    if (!condExpr)
      return true;
    (void)lowerExpr(condExpr, analysisCtx);
    return !analysisCtx.failed;
  };
  if (!collectLoopMutations(ctx, whileStmt->getBody(), analyzeCond,
                            mutatedVars))
    return false;

  LoopSkeleton skeleton;
  if (!buildLoopSkeleton(ctx, mutatedVars, /*hasFirstIterFlag=*/false,
                         mlir::Value(), skeleton))
    return false;
  auto loopOp = skeleton.loop;
  auto *prepareBlock = skeleton.prepareBlock;
  auto *bodyBlock = skeleton.bodyBlock;

  // Prepare region: evaluate loop condition.
  {
    mlir::OpBuilder prepBuilder(ctx.builder.getContext());
    prepBuilder.setInsertionPointToStart(prepareBlock);
    LoweringContext prepCtx(prepBuilder, loc, ctx.returnType, ctx.errorMessage,
                            ctx.sourceManager);
    prepCtx.valueMap = ctx.valueMap;
    prepCtx.loopStack = ctx.loopStack;
    for (auto [index, vd] : llvm::enumerate(mutatedVars))
      prepCtx.valueMap[vd] = prepareBlock->getArgument(index);

    mlir::Value condValue;
    if (condExpr) {
      condValue = lowerExpr(condExpr, prepCtx);
      if (!condValue || prepCtx.failed)
        return false;
    } else {
      condValue = prepBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    }

    llvm::SmallVector<mlir::Value, 8> forwarded;
    forwarded.reserve(mutatedVars.size());
    for (const clang::ValueDecl *vd : mutatedVars)
      forwarded.push_back(prepCtx.valueMap.lookup(vd));

    prepBuilder.create<simt::dialect::ConditionOp>(loc, condValue, forwarded);
  }

  // Body region: execute loop body.
  {
    mlir::OpBuilder bodyBuilder(ctx.builder.getContext());
    bodyBuilder.setInsertionPointToStart(bodyBlock);
    LoweringContext bodyCtx(bodyBuilder, loc, ctx.returnType, ctx.errorMessage,
                            ctx.sourceManager);
    bodyCtx.valueMap = ctx.valueMap;
    bodyCtx.loopStack = ctx.loopStack;
    for (auto [index, vd] : llvm::enumerate(mutatedVars))
      bodyCtx.valueMap[vd] = bodyBlock->getArgument(index);

    if (const clang::Stmt *body = whileStmt->getBody()) {
      if (!lowerStatement(body, bodyCtx) || bodyCtx.failed)
        return false;
    }

    if (!bodyCtx.emittedTerminator) {
      llvm::SmallVector<mlir::Value, 8> yieldOperands;
      yieldOperands.reserve(mutatedVars.size());
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = bodyCtx.valueMap.lookup(vd);
        if (!value)
          value = ctx.valueMap.lookup(vd);
        yieldOperands.push_back(value);
      }
      bodyBuilder.create<simt::dialect::YieldOp>(loc, yieldOperands);
    }
  }

  for (auto [index, vd] : llvm::enumerate(mutatedVars)) {
    ctx.valueMap[vd] = loopOp.getResult(index);
    ctx.mutatedVars.insert(vd);
  }

  ctx.loopStack.pop_back();
  if (!ctx.controlStack.empty() &&
      ctx.controlStack.back().kind == ControlEntryKind::Loop)
    ctx.controlStack.pop_back();

  return true;
}

static bool lowerDoStmt(const clang::DoStmt *doStmt, LoweringContext &ctx) {
  mlir::Location loc = ctx.defaultLoc;

  const clang::Expr *condExpr = doStmt->getCond();
  llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars;
  auto analyzeCond = [&](LoweringContext &analysisCtx) -> bool {
    if (!condExpr)
      return true;
    (void)lowerExpr(condExpr, analysisCtx);
    return !analysisCtx.failed;
  };
  if (!collectLoopMutations(ctx, doStmt->getBody(), analyzeCond, mutatedVars))
    return false;

  mlir::Value firstIterInit =
      ctx.builder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);

  LoopSkeleton skeleton;
  if (!buildLoopSkeleton(ctx, mutatedVars, /*hasFirstIterFlag=*/true,
                         firstIterInit, skeleton))
    return false;
  auto loopOp = skeleton.loop;
  LoopFrame *frame = skeleton.frame;
  auto *prepareBlock = skeleton.prepareBlock;
  auto *bodyBlock = skeleton.bodyBlock;

  // Prepare region: skip condition on first iteration.
  {
    mlir::OpBuilder prepBuilder(ctx.builder.getContext());
    prepBuilder.setInsertionPointToStart(prepareBlock);
    LoweringContext prepCtx(prepBuilder, loc, ctx.returnType, ctx.errorMessage,
                            ctx.sourceManager);
    prepCtx.valueMap = ctx.valueMap;
    prepCtx.loopStack = ctx.loopStack;
    for (auto [index, vd] : llvm::enumerate(mutatedVars))
      prepCtx.valueMap[vd] = prepareBlock->getArgument(index);
    if (!prepCtx.loopStack.empty() && prepCtx.loopStack.back().hasFirstIterFlag)
      prepCtx.loopStack.back().currentFirstIterValue =
          prepareBlock->getArgument(prepCtx.loopStack.back().firstIterIndex);

    mlir::Value condValue;
    if (frame && frame->hasFirstIterFlag) {
      mlir::Value firstIterFlag =
          prepareBlock->getArgument(frame->firstIterIndex);

      llvm::SmallVector<mlir::Type, 8> condResultTypes;
      condResultTypes.push_back(prepBuilder.getI1Type());
      for (const clang::ValueDecl *vd : mutatedVars)
        condResultTypes.push_back(prepCtx.valueMap.lookup(vd).getType());

      auto condIf = prepBuilder.create<simt::dialect::IfOp>(
          loc, condResultTypes, firstIterFlag, /*withElseRegion=*/true);

      // Then region: first iteration, bypass condition.
      {
        auto &thenBlock = condIf.getThenRegion().front();
        mlir::OpBuilder thenBuilder(&thenBlock, thenBlock.end());
        auto trueConst =
            thenBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
        llvm::SmallVector<mlir::Value, 8> thenOperands;
        thenOperands.reserve(condResultTypes.size());
        thenOperands.push_back(trueConst);
        for (const clang::ValueDecl *vd : mutatedVars)
          thenOperands.push_back(prepCtx.valueMap.lookup(vd));
        thenBuilder.create<simt::dialect::YieldOp>(loc, thenOperands);
      }

      // Else region: evaluate condition on subsequent iterations.
      {
        auto &elseBlock = condIf.getElseRegion().front();
        mlir::OpBuilder elseBuilder(&elseBlock, elseBlock.end());
        LoweringContext condCtx(elseBuilder, loc, ctx.returnType,
                                ctx.errorMessage, ctx.sourceManager);
        condCtx.valueMap = prepCtx.valueMap;
        condCtx.loopStack = prepCtx.loopStack;
        condCtx.switchStack = prepCtx.switchStack;
        condCtx.controlStack = prepCtx.controlStack;

        mlir::Value evaluated;
        if (condExpr) {
          evaluated = lowerExpr(condExpr, condCtx);
          if (!evaluated || condCtx.failed)
            return false;
        } else {
          evaluated = elseBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
        }

        llvm::SmallVector<mlir::Value, 8> elseOperands;
        elseOperands.reserve(condResultTypes.size());
        elseOperands.push_back(evaluated);
        for (const clang::ValueDecl *vd : mutatedVars) {
          mlir::Value value = condCtx.valueMap.lookup(vd);
          if (!value)
            value = prepCtx.valueMap.lookup(vd);
          elseOperands.push_back(value);
        }
        elseBuilder.create<simt::dialect::YieldOp>(loc, elseOperands);
      }

      condValue = condIf.getResult(0);
      for (auto [index, vd] : llvm::enumerate(mutatedVars))
        prepCtx.valueMap[vd] = condIf.getResult(index + 1);
    } else {
      if (condExpr) {
        condValue = lowerExpr(condExpr, prepCtx);
        if (!condValue || prepCtx.failed)
          return false;
      } else {
        condValue = prepBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
      }
    }

    llvm::SmallVector<mlir::Value, 8> forwarded;
    forwarded.reserve(mutatedVars.size() + 1);
    for (const clang::ValueDecl *vd : mutatedVars)
      forwarded.push_back(prepCtx.valueMap.lookup(vd));

    mlir::Value nextFlag =
        prepBuilder.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
    forwarded.push_back(nextFlag);
    if (!prepCtx.loopStack.empty())
      prepCtx.loopStack.back().currentFirstIterValue = nextFlag;

    prepBuilder.create<simt::dialect::ConditionOp>(loc, condValue, forwarded);
  }

  // Body region: execute loop body and forward flag state.
  {
    mlir::OpBuilder bodyBuilder(ctx.builder.getContext());
    bodyBuilder.setInsertionPointToStart(bodyBlock);
    LoweringContext bodyCtx(bodyBuilder, loc, ctx.returnType, ctx.errorMessage,
                            ctx.sourceManager);
    bodyCtx.valueMap = ctx.valueMap;
    bodyCtx.loopStack = ctx.loopStack;
    bodyCtx.switchStack = ctx.switchStack;
    bodyCtx.controlStack = ctx.controlStack;
    for (auto [index, vd] : llvm::enumerate(mutatedVars))
      bodyCtx.valueMap[vd] = bodyBlock->getArgument(index);

    mlir::Value continueFlag =
        bodyBuilder.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
    if (!bodyCtx.loopStack.empty() && bodyCtx.loopStack.back().hasFirstIterFlag)
      bodyCtx.loopStack.back().currentFirstIterValue = continueFlag;

    if (const clang::Stmt *body = doStmt->getBody()) {
      if (!lowerStatement(body, bodyCtx) || bodyCtx.failed)
        return false;
    }

    if (!bodyCtx.emittedTerminator) {
      llvm::SmallVector<mlir::Value, 8> yieldOperands;
      yieldOperands.reserve(mutatedVars.size() + 1);
      for (const clang::ValueDecl *vd : mutatedVars) {
        mlir::Value value = bodyCtx.valueMap.lookup(vd);
        if (!value)
          value = ctx.valueMap.lookup(vd);
        yieldOperands.push_back(value);
      }
      yieldOperands.push_back(continueFlag);
      bodyBuilder.create<simt::dialect::YieldOp>(loc, yieldOperands);
    }
  }

  for (auto [index, vd] : llvm::enumerate(mutatedVars)) {
    ctx.valueMap[vd] = loopOp.getResult(index);
    ctx.mutatedVars.insert(vd);
  }

  ctx.loopStack.pop_back();
  if (!ctx.controlStack.empty() &&
      ctx.controlStack.back().kind == ControlEntryKind::Loop)
    ctx.controlStack.pop_back();

  return true;
}

static bool lowerSwitchStmt(const clang::SwitchStmt *switchStmt,
                            LoweringContext &ctx) {
  mlir::Location loc = ctx.defaultLoc;

  if (const clang::Stmt *init = switchStmt->getInit()) {
    if (!lowerStatement(init, ctx) || ctx.failed)
      return false;
  }

  if (switchStmt->getConditionVariable())
    return ctx.fail("switch condition variables are not supported");

  struct CaseInfo {
    const clang::SwitchCase *label = nullptr;
    llvm::SmallVector<const clang::Stmt *, 8> statements;
  };

  llvm::SmallVector<CaseInfo, 8> cases;

  std::function<bool(CaseInfo *, const clang::Stmt *)> addStatement;
  addStatement = [&](CaseInfo *current, const clang::Stmt *stmt) -> bool {
    if (!stmt)
      return true;
    if (llvm::isa<clang::NullStmt>(stmt))
      return true;
    if (const auto *attr = llvm::dyn_cast<clang::AttributedStmt>(stmt))
      return addStatement(current, attr->getSubStmt());
    if (!current)
      return ctx.fail("statement outside of switch cases is not supported"),
             false;
    current->statements.push_back(stmt);
    return true;
  };

  auto pushCase = [&](const clang::SwitchCase *sc) -> CaseInfo * {
    cases.push_back({sc, {}});
    return &cases.back();
  };

  const clang::Stmt *body = switchStmt->getBody();
  llvm::SmallVector<const clang::Stmt *, 8> topLevel;
  if (const auto *compound = llvm::dyn_cast<clang::CompoundStmt>(body)) {
    topLevel.append(compound->body_begin(), compound->body_end());
  } else if (body) {
    topLevel.push_back(body);
  }

  CaseInfo *current = nullptr;
  for (const clang::Stmt *child : topLevel) {
    if (const auto *sc = llvm::dyn_cast<clang::SwitchCase>(child)) {
      const clang::SwitchCase *active = sc;
      const clang::Stmt *sub = nullptr;
      do {
        current = pushCase(active);
        sub = active->getSubStmt();
        active = llvm::dyn_cast<clang::SwitchCase>(sub);
      } while (active);
      if (sub && !addStatement(current, sub))
        return false;
    } else {
      if (!addStatement(current, child))
        return false;
    }
  }

  mlir::Value selector = lowerExpr(switchStmt->getCond(), ctx);
  if (!selector)
    return false;

  llvm::SmallPtrSet<const clang::ValueDecl *, 8> mutatedSet;
  for (const CaseInfo &info : cases) {
    mlir::Region analysisRegion;
    analysisRegion.emplaceBlock();
    mlir::OpBuilder analysisBuilder(ctx.builder.getContext());
    analysisBuilder.setInsertionPointToStart(&analysisRegion.front());
    LoweringContext analysisCtx(analysisBuilder, loc, ctx.returnType,
                                ctx.errorMessage, ctx.sourceManager);
    analysisCtx.valueMap = ctx.valueMap;
    analysisCtx.loopStack = ctx.loopStack;
    analysisCtx.switchStack = ctx.switchStack;
    analysisCtx.controlStack = ctx.controlStack;
    SwitchFrame analysisFrame;
    analysisFrame.analysisOnly = true;
    analysisCtx.switchStack.push_back(analysisFrame);
    analysisCtx.controlStack.push_back(
        {ControlEntryKind::Switch, analysisCtx.switchStack.size() - 1});

    for (const clang::Stmt *caseStmt : info.statements) {
      if (!lowerStatement(caseStmt, analysisCtx) || analysisCtx.failed)
        return false;
      if (analysisCtx.emittedTerminator)
        break;
    }

    analysisCtx.switchStack.pop_back();
    if (!analysisCtx.controlStack.empty() &&
        analysisCtx.controlStack.back().kind == ControlEntryKind::Switch)
      analysisCtx.controlStack.pop_back();
    mutatedSet.insert(analysisCtx.mutatedVars.begin(),
                      analysisCtx.mutatedVars.end());
  }

  llvm::SmallVector<const clang::ValueDecl *, 8> mutatedVars(mutatedSet.begin(),
                                                             mutatedSet.end());
  llvm::sort(mutatedVars,
             [](const clang::ValueDecl *lhs, const clang::ValueDecl *rhs) {
               return lhs < rhs;
             });

  llvm::SmallVector<mlir::Value, 8> currentValues;
  currentValues.reserve(mutatedVars.size());
  for (const clang::ValueDecl *vd : mutatedVars) {
    mlir::Value initial = ctx.valueMap.lookup(vd);
    if (!initial)
      return ctx.fail("reference to unknown switch variable");
    currentValues.push_back(initial);
  }

  mlir::Value boolZero =
      ctx.builder.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
  mlir::Value currentHasMatched = boolZero;
  mlir::Value currentExecuting = boolZero;
  mlir::Value currentCompleted = boolZero;

  for (const CaseInfo &info : cases) {
    mlir::Value notCompleted = ctx.builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, currentCompleted, boolZero);
    mlir::Value hasNotMatched = ctx.builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, currentHasMatched, boolZero);

    mlir::Value caseMatch;
    if (const auto *caseStmt = llvm::dyn_cast<clang::CaseStmt>(info.label)) {
      mlir::Value caseValue = lowerExpr(caseStmt->getLHS(), ctx);
      if (!caseValue)
        return false;
      mlir::Value valueEquals = ctx.builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, selector, caseValue);
      mlir::Value available = ctx.builder.create<mlir::arith::AndIOp>(
          loc, hasNotMatched, notCompleted);
      caseMatch =
          ctx.builder.create<mlir::arith::AndIOp>(loc, available, valueEquals);
    } else {
      caseMatch = ctx.builder.create<mlir::arith::AndIOp>(loc, hasNotMatched,
                                                          notCompleted);
    }

    mlir::Value executeCondition = ctx.builder.create<mlir::arith::OrIOp>(
        loc, currentExecuting, caseMatch);
    mlir::Value enterCase = ctx.builder.create<mlir::arith::AndIOp>(
        loc, executeCondition, notCompleted);

    llvm::SmallVector<mlir::Type, 8> resultTypes;
    resultTypes.reserve(mutatedVars.size() + 3);
    for (mlir::Value value : currentValues)
      resultTypes.push_back(value.getType());
    resultTypes.push_back(ctx.builder.getI1Type());
    resultTypes.push_back(ctx.builder.getI1Type());
    resultTypes.push_back(ctx.builder.getI1Type());

    auto ifOp =
        ctx.builder.create<simt::dialect::IfOp>(loc, resultTypes, enterCase,
                                                /*withElseRegion=*/true);

    auto &thenBlock = ifOp.getThenRegion().front();
    thenBlock.clear();
    mlir::OpBuilder thenBuilder(ctx.builder.getContext());
    thenBuilder.setInsertionPointToStart(&thenBlock);

    LoweringContext caseCtx(thenBuilder, loc, ctx.returnType, ctx.errorMessage,
                            ctx.sourceManager);
    caseCtx.valueMap = ctx.valueMap;
    caseCtx.loopStack = ctx.loopStack;
    caseCtx.switchStack = ctx.switchStack;
    caseCtx.controlStack = ctx.controlStack;
    for (auto [vd, value] : llvm::zip(mutatedVars, currentValues))
      caseCtx.valueMap[vd] = value;

    SwitchFrame frame;
    frame.carriedVars = mutatedVars;
    frame.hasMatchedIndex = mutatedVars.size();
    frame.executingIndex = mutatedVars.size() + 1;
    frame.completedIndex = mutatedVars.size() + 2;
    frame.initialValues.assign(currentValues.begin(), currentValues.end());
    frame.breakHasMatchedValue =
        thenBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    frame.breakExecutingValue =
        thenBuilder.create<mlir::arith::ConstantIntOp>(loc, 0, 1);
    frame.breakCompletedValue =
        thenBuilder.create<mlir::arith::ConstantIntOp>(loc, 1, 1);
    caseCtx.switchStack.push_back(frame);
    caseCtx.controlStack.push_back(
        {ControlEntryKind::Switch, caseCtx.switchStack.size() - 1});

    for (const clang::Stmt *caseStmt : info.statements) {
      if (!lowerStatement(caseStmt, caseCtx) || caseCtx.failed)
        return false;
      if (caseCtx.emittedTerminator)
        break;
    }

    if (!caseCtx.emittedTerminator) {
      llvm::SmallVector<mlir::Value, 8> yieldValues;
      yieldValues.reserve(mutatedVars.size() + 3);
      for (auto [index, vd] : llvm::enumerate(mutatedVars)) {
        mlir::Value value = caseCtx.valueMap.lookup(vd);
        if (!value && index < currentValues.size())
          value = currentValues[index];
        if (!value) {
          caseCtx.fail("switch case missing value for variable");
          return false;
        }
        yieldValues.push_back(value);
        caseCtx.mutatedVars.insert(vd);
      }
      mlir::Value updatedHasMatched = thenBuilder.create<mlir::arith::OrIOp>(
          loc, currentHasMatched, caseMatch);
      mlir::Value updatedExecuting = thenBuilder.create<mlir::arith::OrIOp>(
          loc, currentExecuting, caseMatch);
      yieldValues.push_back(updatedHasMatched);
      yieldValues.push_back(updatedExecuting);
      yieldValues.push_back(currentCompleted);
      thenBuilder.create<simt::dialect::YieldOp>(loc, yieldValues);
    }

    caseCtx.switchStack.pop_back();
    if (!caseCtx.controlStack.empty() &&
        caseCtx.controlStack.back().kind == ControlEntryKind::Switch)
      caseCtx.controlStack.pop_back();
    ctx.mutatedVars.insert(caseCtx.mutatedVars.begin(),
                           caseCtx.mutatedVars.end());

    auto &elseBlock = ifOp.getElseRegion().front();
    elseBlock.clear();
    mlir::OpBuilder elseBuilder(ctx.builder.getContext());
    elseBuilder.setInsertionPointToStart(&elseBlock);

    llvm::SmallVector<mlir::Value, 8> elseValues = currentValues;
    elseValues.push_back(currentHasMatched);
    elseValues.push_back(currentExecuting);
    elseValues.push_back(currentCompleted);
    elseBuilder.create<simt::dialect::YieldOp>(loc, elseValues);

    currentValues.clear();
    currentValues.reserve(mutatedVars.size());
    for (size_t index = 0; index < mutatedVars.size(); ++index)
      currentValues.push_back(ifOp.getResult(index));
    currentHasMatched = ifOp.getResult(mutatedVars.size());
    currentExecuting = ifOp.getResult(mutatedVars.size() + 1);
    currentCompleted = ifOp.getResult(mutatedVars.size() + 2);
  }

  for (auto [vd, value] : llvm::zip(mutatedVars, currentValues))
    ctx.valueMap[vd] = value;

  ctx.mutatedVars.insert(mutatedVars.begin(), mutatedVars.end());

  return true;
}

class FunctionLoweringVisitor
    : public clang::RecursiveASTVisitor<FunctionLoweringVisitor> {
public:
  FunctionLoweringVisitor(mlir::OwningOpRef<mlir::ModuleOp> &module,
                          mlir::OpBuilder &builder)
      : module(module), moduleBuilder(builder) {}

  bool VisitVarDecl(const clang::VarDecl *decl) {
    if (!decl->hasGlobalStorage() || decl->isStaticLocal())
      return true;
    if (decl->isImplicit())
      return true;

    mlir::Type type = convertType(decl->getType(), moduleBuilder);
    if (!type) {
      clang::LangAS addressSpace = decl->getType().getAddressSpace();
      if (addressSpace == clang::LangAS::hlsl_groupshared) {
        clang::QualType varType = decl->getType();
        const clang::Type *elementTy =
            varType.getCanonicalType().getTypePtrOrNull();
        if (!elementTy)
          return true;
        while (auto *arrayTy = llvm::dyn_cast<clang::ArrayType>(elementTy))
          elementTy =
              arrayTy->getElementType().getCanonicalType().getTypePtr();
        mlir::Type elementType = convertType(clang::QualType(elementTy, 0),
                                             moduleBuilder);
        if (!elementType)
          return true;
        type = simt::dialect::ResourceType::get(
            moduleBuilder.getContext(),
            simt::dialect::MemorySpace::Shared, elementType);
      }
    }
    if (!type)
      return true;

    if (!mlir::isa<simt::dialect::ResourceType>(type))
      return true;

    if (resourceSet.insert(decl).second)
      resourceDecls.push_back(decl);
    return true;
  }

  bool VisitFunctionDecl(const clang::FunctionDecl *decl) {
    const auto *shaderAttr = decl->getAttr<clang::HLSLShaderAttr>();
    if (!shaderAttr || shaderAttr->getType() != llvm::Triple::Compute)
      return true;

    foundComputeShader = true;

    auto name = decl->getNameAsString();
    mlir::Location loc = moduleBuilder.getUnknownLoc();

    mlir::OpBuilder::InsertionGuard guard(moduleBuilder);
    moduleBuilder.setInsertionPointToEnd(module->getBody());

    llvm::SmallVector<mlir::Type> argTypes;
    argTypes.reserve(decl->getNumParams() + resourceDecls.size());
    for (const clang::ParmVarDecl *param : decl->parameters()) {
      mlir::Type type = convertType(param->getType(), moduleBuilder);
      if (!type || mlir::isa<mlir::NoneType>(type)) {
        recordError("unsupported parameter type in function '" + name + "'");
        return false;
      }
      argTypes.push_back(type);
    }

    for (const clang::VarDecl *resourceDecl : resourceDecls) {
      mlir::Type type = convertType(resourceDecl->getType(), moduleBuilder);
      if (!type || !mlir::isa<simt::dialect::ResourceType>(type)) {
        recordError("unsupported resource type in function '" + name + "'");
        return false;
      }
      argTypes.push_back(type);
    }

    mlir::Type resultType = convertType(decl->getReturnType(), moduleBuilder);
    if (!resultType)
      resultType = moduleBuilder.getNoneType();

    llvm::SmallVector<mlir::Type> resultTypes;
    if (!mlir::isa<mlir::NoneType>(resultType))
      resultTypes.push_back(resultType);

    auto funcType = moduleBuilder.getFunctionType(argTypes, resultTypes);
    auto func = moduleBuilder.create<mlir::func::FuncOp>(loc, name, funcType);

    if (const auto *numThreads = decl->getAttr<clang::HLSLNumThreadsAttr>()) {
      llvm::SmallVector<int64_t, 3> dims = {
          static_cast<int64_t>(numThreads->getX()),
          static_cast<int64_t>(numThreads->getY()),
          static_cast<int64_t>(numThreads->getZ())};
      auto attr = mlir::DenseI64ArrayAttr::get(func.getContext(), dims);
      func->setAttr("simt.num_threads", attr);
    }

    mlir::Block *entry = func.addEntryBlock();
    mlir::OpBuilder funcBuilder(entry, entry->begin());
    funcBuilder.create<simt::dialect::ActiveMaskOp>(loc,
                                                    funcBuilder.getI64Type());

    const clang::SourceManager &sourceManager =
        decl->getASTContext().getSourceManager();

    LoweringContext ctx(funcBuilder, loc,
                        resultTypes.empty() ? mlir::Type()
                                            : resultTypes.front(),
                        errorMessage, &sourceManager);
    auto entryArgs = entry->getArguments();
    size_t paramCount = decl->getNumParams();
    for (size_t index = 0; index < paramCount; ++index)
      ctx.valueMap[decl->getParamDecl(index)] = entryArgs[index];
    for (auto [resourceDecl, arg] :
         llvm::zip(resourceDecls, entryArgs.drop_front(paramCount)))
      ctx.valueMap[resourceDecl] = arg;

    const clang::Stmt *body = decl->getBody();
    if (const auto *compound = llvm::dyn_cast<clang::CompoundStmt>(body))
      lowerCompoundStmt(compound, ctx);
    else if (body)
      lowerStatement(body, ctx);

    if (ctx.failed)
      return false;

    if (!ctx.emittedTerminator) {
      bool expectsValue = static_cast<bool>(ctx.returnType) &&
                          !mlir::isa<mlir::NoneType>(ctx.returnType);
      if (expectsValue) {
        mlir::Value zero = buildZeroValue(ctx, ctx.returnType);
        if (!zero)
          return false;
        ctx.builder.create<mlir::func::ReturnOp>(ctx.builder.getUnknownLoc(),
                                                 zero);
      } else {
        ctx.builder.create<mlir::func::ReturnOp>(ctx.builder.getUnknownLoc());
      }
    }

    return true;
  }

  bool encounteredError() const { return !errorMessage.empty(); }
  const std::string &error() const { return errorMessage; }
  bool hasComputeShader() const { return foundComputeShader; }

private:
  void recordError(const std::string &msg) {
    if (errorMessage.empty())
      errorMessage = msg;
  }

  mlir::OwningOpRef<mlir::ModuleOp> &module;
  mlir::OpBuilder &moduleBuilder;
  std::string errorMessage;
  bool foundComputeShader = false;
  llvm::SmallVector<const clang::VarDecl *, 8> resourceDecls;
  llvm::SmallPtrSet<const clang::VarDecl *, 8> resourceSet;
};

class TranslationASTConsumer : public clang::ASTConsumer {
public:
  explicit TranslationASTConsumer(FunctionLoweringVisitor &visitor)
      : visitor(visitor) {}

  void HandleTranslationUnit(clang::ASTContext &context) override {
    visitor.TraverseDecl(context.getTranslationUnitDecl());
  }

private:
  FunctionLoweringVisitor &visitor;
};

class TranslationFrontendAction : public clang::ASTFrontendAction {
public:
  explicit TranslationFrontendAction(FunctionLoweringVisitor &visitor)
      : visitor(visitor) {}

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) override {
    return std::make_unique<TranslationASTConsumer>(visitor);
  }

private:
  FunctionLoweringVisitor &visitor;
};

} // namespace

Result<mlir::OwningOpRef<mlir::ModuleOp>>
translateComputeShader(mlir::MLIRContext &context, llvm::StringRef fileName,
                       llvm::StringRef source,
                       const TranslationOptions &options) {
  context.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                      mlir::vector::VectorDialect,
                      simt::dialect::SimtStepDialect>();

  mlir::OpBuilder builder(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module->getBody());

  FunctionLoweringVisitor visitor(module, builder);
  std::vector<std::string> clangArgs = {"-x", "hlsl", "-std=hlsl2021",
                                        "-D__HLSL__"};

  if (auto triple = buildDxilTripleForProfile(options.shaderProfile)) {
    clangArgs.emplace_back("-target");
    clangArgs.emplace_back(std::move(*triple));
  } else {
    return Result<mlir::OwningOpRef<mlir::ModuleOp>>::err(
        "unsupported shader profile '" + options.shaderProfile + "'");
  }

  clangArgs.emplace_back("-Xclang");
  clangArgs.emplace_back("-finclude-default-header");
  clangArgs.emplace_back("-Wno-hlsl-dxc-compatability");

  for (const std::string &include : options.forcedIncludeFiles) {
    clangArgs.emplace_back("-include");
    clangArgs.emplace_back(include);
  }

  for (const std::string &dir : options.extraIncludeDirs) {
    clangArgs.emplace_back("-isystem");
    clangArgs.emplace_back(dir);
  }

  if (!options.resourceDir.empty()) {
    clangArgs.emplace_back("-resource-dir");
    clangArgs.emplace_back(options.resourceDir);
  }

  auto action = std::make_unique<TranslationFrontendAction>(visitor);
  if (!clang::tooling::runToolOnCodeWithArgs(std::move(action), source.str(),
                                             clangArgs, fileName.str()))
    return Result<mlir::OwningOpRef<mlir::ModuleOp>>::err(
        "failed to translate HLSL input");

  if (visitor.encounteredError())
    return Result<mlir::OwningOpRef<mlir::ModuleOp>>::err(visitor.error());

  if (!visitor.hasComputeShader())
    return Result<mlir::OwningOpRef<mlir::ModuleOp>>::err(
        "no compute shader entry point found");

  return Result<mlir::OwningOpRef<mlir::ModuleOp>>::ok(std::move(module));
}

} // namespace simt_hlsl_import
