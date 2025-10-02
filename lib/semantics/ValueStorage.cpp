#include "simt-step/semantics/ValueStorage.h"

#include <cmath>

#include <mlir/IR/BuiltinTypes.h>

namespace simt::semantics {

ScalarValue::ScalarValue() : kind_(ValueKind::Int32) { data_.i32 = 0; }

ScalarValue ScalarValue::fromBool(bool v) {
    ScalarValue value;
    value.kind_ = ValueKind::Bool;
    value.data_.b = v;
    return value;
}

ScalarValue ScalarValue::fromInt(int32_t v) {
    ScalarValue value;
    value.kind_ = ValueKind::Int32;
    value.data_.i32 = v;
    return value;
}

ScalarValue ScalarValue::fromFloat(float v) {
    ScalarValue value;
    value.kind_ = ValueKind::Float32;
    value.data_.f32 = v;
    return value;
}

bool ScalarValue::asBool() const {
    switch (kind_) {
    case ValueKind::Bool:
        return data_.b;
    case ValueKind::Int32:
        return data_.i32 != 0;
    case ValueKind::Float32:
        return data_.f32 != 0.0f;
    }
    return false;
}

int32_t ScalarValue::asInt() const {
    switch (kind_) {
    case ValueKind::Bool:
        return data_.b ? 1 : 0;
    case ValueKind::Int32:
        return data_.i32;
    case ValueKind::Float32:
        return static_cast<int32_t>(std::lround(data_.f32));
    }
    return 0;
}

float ScalarValue::asFloat() const {
    switch (kind_) {
    case ValueKind::Bool:
        return data_.b ? 1.0f : 0.0f;
    case ValueKind::Int32:
        return static_cast<float>(data_.i32);
    case ValueKind::Float32:
        return data_.f32;
    }
    return 0.0f;
}

std::optional<ValueKind> classifyType(mlir::Type type) {
    if (!type)
        return std::nullopt;

    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(type)) {
        if (intTy.getWidth() == 1)
            return ValueKind::Bool;
        if (intTy.getWidth() == 32 && intTy.isSignless())
            return ValueKind::Int32;
    } else if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(type)) {
        if (floatTy.isF32())
            return ValueKind::Float32;
    }

    return std::nullopt;
}

static_assert(makeLaneIdLut<4>()[3] == 3,
              "lane id LUT must produce identity mapping");

} // namespace simt::semantics
