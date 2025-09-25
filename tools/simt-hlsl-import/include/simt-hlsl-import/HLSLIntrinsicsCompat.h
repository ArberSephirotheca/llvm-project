#ifndef SIMT_HLSL_INTRINSICS_COMPAT_H
#define SIMT_HLSL_INTRINSICS_COMPAT_H

// Minimal HLSL intrinsic declarations to unblock Clang parsing when the
// toolchain headers do not yet provide them. When the upstream alias header is
// available we include it to avoid redeclaring existing intrinsics.

#if __has_include(<hlsl/hlsl_alias_intrinsics.h>)
#include <hlsl/hlsl_alias_intrinsics.h>
#define SIMT_COMPAT_HAS_ALIAS 1
#else
#define SIMT_COMPAT_HAS_ALIAS 0
#endif

namespace hlsl {

// Interlocked atomics -------------------------------------------------------

void InterlockedAdd(inout int dest, int value);
void InterlockedAdd(inout int dest, int value, out int original);
void InterlockedExchange(inout int dest, int value);
void InterlockedExchange(inout int dest, int value, out int original);

void InterlockedCompareExchange(inout int dest, int compareValue, int value);
void InterlockedCompareExchange(inout int dest, int compareValue, int value,
                                out int original);

void InterlockedMin(inout int dest, int value);
void InterlockedMin(inout int dest, int value, out int original);

void InterlockedMax(inout int dest, int value);
void InterlockedMax(inout int dest, int value, out int original);

void InterlockedAnd(inout int dest, int value);
void InterlockedAnd(inout int dest, int value, out int original);

void InterlockedOr(inout int dest, int value);
void InterlockedOr(inout int dest, int value, out int original);

void InterlockedXor(inout int dest, int value);
void InterlockedXor(inout int dest, int value, out int original);

// Wave intrinsics -----------------------------------------------------------

#if SIMT_COMPAT_HAS_ALIAS == 0
bool WaveActiveAllTrue(bool value);
bool WaveActiveAnyTrue(bool value);
unsigned int WaveGetLaneIndex();
#endif

// Memory barriers -----------------------------------------------------------

void GroupMemoryBarrier();
void GroupMemoryBarrierWithGroupSync();
void DeviceMemoryBarrier();
void DeviceMemoryBarrierWithGroupSync();
void AllMemoryBarrier();
void AllMemoryBarrierWithGroupSync();

#if SIMT_COMPAT_HAS_ALIAS == 0
// Fallback declarations for additional barrier helpers that may be missing
// from older toolchains can be added here when needed.
#endif

} // namespace hlsl

#undef SIMT_COMPAT_HAS_ALIAS

#endif // SIMT_HLSL_INTRINSICS_COMPAT_H
