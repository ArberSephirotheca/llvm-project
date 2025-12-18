#pragma once

#include <cassert>
#include <cstdint>
#include <optional>
#include <variant>

#include <mlir/IR/Types.h>
#include <mlir/IR/BuiltinTypes.h>

namespace simt::semantics {

/// Scalar value domain interpreted by the CPS engine.
class SemValue {
public:
    enum class Kind { None, Bool, Int32, Int64, Float32 };

    SemValue() = default;

    static SemValue fromBool(bool v) {
        SemValue value;
        value.storage_ = v;
        return value;
    }

    static SemValue fromInt32(int32_t v) {
        SemValue value;
        value.storage_ = v;
        return value;
    }

    static SemValue fromInt64(int64_t v) {
        SemValue value;
        value.storage_ = v;
        return value;
    }

    static SemValue fromFloat(float v) {
        SemValue value;
        value.storage_ = v;
        return value;
    }

    Kind kind() const {
        if (std::holds_alternative<bool>(storage_))
            return Kind::Bool;
        if (std::holds_alternative<int32_t>(storage_))
            return Kind::Int32;
        if (std::holds_alternative<int64_t>(storage_))
            return Kind::Int64;
        if (std::holds_alternative<float>(storage_))
            return Kind::Float32;
        return Kind::None;
    }

    bool isBool() const { return kind() == Kind::Bool; }
    bool isInt32() const { return kind() == Kind::Int32; }
    bool isInt64() const { return kind() == Kind::Int64; }
    bool isFloat32() const { return kind() == Kind::Float32; }
    bool isInteger() const { return isBool() || isInt32() || isInt64(); }

    bool isNone() const { return kind() == Kind::None; }

    bool asBool() const {
        if (std::holds_alternative<bool>(storage_))
            return std::get<bool>(storage_);
        if (std::holds_alternative<int32_t>(storage_))
            return std::get<int32_t>(storage_) != 0;
        if (std::holds_alternative<int64_t>(storage_))
            return std::get<int64_t>(storage_) != 0;
        if (std::holds_alternative<float>(storage_))
            return std::get<float>(storage_) != 0.0f;
        return false;
    }

    int64_t asInt64() const {
        if (std::holds_alternative<int32_t>(storage_))
            return static_cast<int64_t>(std::get<int32_t>(storage_));
        if (std::holds_alternative<int64_t>(storage_))
            return std::get<int64_t>(storage_);
        if (std::holds_alternative<bool>(storage_))
            return std::get<bool>(storage_) ? 1 : 0;
        if (std::holds_alternative<float>(storage_))
            return static_cast<int64_t>(std::get<float>(storage_));
        return 0;
    }

    float asFloat32() const {
        if (std::holds_alternative<float>(storage_))
            return std::get<float>(storage_);
        if (std::holds_alternative<int32_t>(storage_))
            return static_cast<float>(std::get<int32_t>(storage_));
        if (std::holds_alternative<int64_t>(storage_))
            return static_cast<float>(std::get<int64_t>(storage_));
        if (std::holds_alternative<bool>(storage_))
            return std::get<bool>(storage_) ? 1.0f : 0.0f;
        return 0.0f;
    }

    double asFloat64() const {
        if (std::holds_alternative<float>(storage_))
            return static_cast<double>(std::get<float>(storage_));
        if (std::holds_alternative<int32_t>(storage_))
            return static_cast<double>(std::get<int32_t>(storage_));
        if (std::holds_alternative<int64_t>(storage_))
            return static_cast<double>(std::get<int64_t>(storage_));
        if (std::holds_alternative<bool>(storage_))
            return std::get<bool>(storage_) ? 1.0 : 0.0;
        return 0.0;
    }

    SemValue neg() const {
        if (isFloat32())
            return SemValue::fromFloat(-asFloat32());
        return SemValue::fromInt64(-asInt64());
    }

    SemValue add(const SemValue &rhs) const {
        Kind result = arithmeticKind(*this, rhs);
        if (result == Kind::Float32)
            return SemValue::fromFloat(static_cast<float>(asFloat64() + rhs.asFloat64()));
        if (result == Kind::Int64)
            return SemValue::fromInt64(asInt64() + rhs.asInt64());
        return SemValue::fromInt32(static_cast<int32_t>(asInt64() + rhs.asInt64()));
    }

    SemValue sub(const SemValue &rhs) const {
        Kind result = arithmeticKind(*this, rhs);
        if (result == Kind::Float32)
            return SemValue::fromFloat(static_cast<float>(asFloat64() - rhs.asFloat64()));
        if (result == Kind::Int64)
            return SemValue::fromInt64(asInt64() - rhs.asInt64());
        return SemValue::fromInt32(static_cast<int32_t>(asInt64() - rhs.asInt64()));
    }

    SemValue mul(const SemValue &rhs) const {
        Kind result = arithmeticKind(*this, rhs);
        if (result == Kind::Float32)
            return SemValue::fromFloat(static_cast<float>(asFloat64() * rhs.asFloat64()));
        if (result == Kind::Int64)
            return SemValue::fromInt64(asInt64() * rhs.asInt64());
        return SemValue::fromInt32(static_cast<int32_t>(asInt64() * rhs.asInt64()));
    }

    SemValue div(const SemValue &rhs) const {
        Kind result = arithmeticKind(*this, rhs);
        if (result == Kind::Float32) {
            return SemValue::fromFloat(static_cast<float>(asFloat64() / rhs.asFloat64()));
        }
        int64_t divisor = rhs.asInt64();
        assert(divisor != 0 && "division by zero");
        if (result == Kind::Int64)
            return SemValue::fromInt64(asInt64() / divisor);
        return SemValue::fromInt32(static_cast<int32_t>(asInt64() / divisor));
    }

    SemValue logicalAnd(const SemValue &rhs) const {
        return SemValue::fromBool(asBool() && rhs.asBool());
    }

    SemValue logicalOr(const SemValue &rhs) const {
        return SemValue::fromBool(asBool() || rhs.asBool());
    }

    SemValue logicalNot() const { return SemValue::fromBool(!asBool()); }

    SemValue bitAnd(const SemValue &rhs) const {
        assert(isInteger() && rhs.isInteger() && "bitAnd requires integers");
        Kind result = arithmeticKind(*this, rhs);
        if (result == Kind::Int64)
            return SemValue::fromInt64(asInt64() & rhs.asInt64());
        return SemValue::fromInt32(
            static_cast<int32_t>(asInt64() & rhs.asInt64()));
    }

    SemValue bitOr(const SemValue &rhs) const {
        assert(isInteger() && rhs.isInteger() && "bitOr requires integers");
        Kind result = arithmeticKind(*this, rhs);
        if (result == Kind::Int64)
            return SemValue::fromInt64(asInt64() | rhs.asInt64());
        return SemValue::fromInt32(
            static_cast<int32_t>(asInt64() | rhs.asInt64()));
    }

    SemValue cmpEqual(const SemValue &rhs) const {
        Kind result = arithmeticKind(*this, rhs);
        if (result == Kind::Float32)
            return SemValue::fromBool(asFloat64() == rhs.asFloat64());
        return SemValue::fromBool(asInt64() == rhs.asInt64());
    }

    SemValue cmpNotEqual(const SemValue &rhs) const {
        return SemValue::fromBool(!cmpEqual(rhs).asBool());
    }

    SemValue cmpLess(const SemValue &rhs) const {
        Kind result = arithmeticKind(*this, rhs);
        if (result == Kind::Float32)
            return SemValue::fromBool(asFloat64() < rhs.asFloat64());
        return SemValue::fromBool(asInt64() < rhs.asInt64());
    }

    SemValue cmpLessEqual(const SemValue &rhs) const {
        Kind result = arithmeticKind(*this, rhs);
        if (result == Kind::Float32)
            return SemValue::fromBool(asFloat64() <= rhs.asFloat64());
        return SemValue::fromBool(asInt64() <= rhs.asInt64());
    }

    SemValue cmpGreater(const SemValue &rhs) const {
        return SemValue::fromBool(!cmpLessEqual(rhs).asBool());
    }

    SemValue cmpGreaterEqual(const SemValue &rhs) const {
        return SemValue::fromBool(!cmpLess(rhs).asBool());
    }

    static std::optional<SemValue> zeroForType(mlir::Type type) {
        if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type)) {
            if (intTy.getWidth() <= 32)
                return SemValue::fromInt32(0);
            return SemValue::fromInt64(0);
        }
        if (mlir::isa<mlir::Float32Type>(type))
            return SemValue::fromFloat(0.0f);
        if (mlir::isa<mlir::Float16Type>(type) || mlir::isa<mlir::BFloat16Type>(type))
            return SemValue::fromFloat(0.0f);
        if (mlir::isa<mlir::IndexType>(type))
            return SemValue::fromInt64(0);
        if (mlir::isa<mlir::NoneType>(type))
            return SemValue();
        return std::nullopt;
    }

private:
    static Kind canonical(Kind kind) {
        if (kind == Kind::Bool)
            return Kind::Int32;
        return kind;
    }

    static Kind arithmeticKind(const SemValue &lhs, const SemValue &rhs) {
        Kind l = canonical(lhs.kind());
        Kind r = canonical(rhs.kind());
        if (l == Kind::Float32 || r == Kind::Float32)
            return Kind::Float32;
        if (l == Kind::Int64 || r == Kind::Int64)
            return Kind::Int64;
        return Kind::Int32;
    }

    std::variant<std::monostate, bool, int32_t, int64_t, float> storage_;
};

} // namespace simt::semantics
