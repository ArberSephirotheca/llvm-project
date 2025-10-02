#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <mlir/IR/Types.h>

namespace simt::semantics {

/// Identifies the concrete scalar payload held by a lane.
enum class ValueKind { Bool, Int32, Float32 };

class ScalarValue {
public:
    ScalarValue();

    static ScalarValue fromBool(bool v);
    static ScalarValue fromInt(int32_t v);
    static ScalarValue fromFloat(float v);

    ValueKind getKind() const { return kind_; }

    bool asBool() const;
    int32_t asInt() const;
    float asFloat() const;

private:
    ValueKind kind_;
    union {
        bool b;
        int32_t i32;
        float f32;
    } data_;
};

/// Structure-of-arrays container holding one scalar per lane.
template <std::size_t WaveWidth>
class LaneValueSlice {
public:
    constexpr ScalarValue &operator[](std::size_t lane) { return values_[lane]; }
    constexpr const ScalarValue &operator[](std::size_t lane) const { return values_[lane]; }
    static constexpr std::size_t size() { return WaveWidth; }

private:
    std::array<ScalarValue, WaveWidth> values_{};
};

/// Compile-time helper that produces the lane-id lookup table for a given wave.
template <std::size_t WaveWidth>
consteval std::array<std::uint32_t, WaveWidth> makeLaneIdLut() {
    std::array<std::uint32_t, WaveWidth> lut{};
    for (std::uint32_t lane = 0; lane < WaveWidth; ++lane)
        lut[lane] = lane;
    return lut;
}

std::optional<ValueKind> classifyType(mlir::Type type);

} // namespace simt::semantics

