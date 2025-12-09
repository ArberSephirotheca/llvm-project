#include "SimtParticipantPatterns.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace mlir;

namespace simt::fuzz {
namespace patterns {

Value laneEquals(OpBuilder &builder, Location loc, Value tid, std::int64_t laneIdx) {
    auto c = builder.create<arith::ConstantIntOp>(loc, laneIdx, 32);
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tid, c);
}

Value parity(OpBuilder &builder, Location loc, Value tid, bool even) {
    auto one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    auto bit = builder.create<arith::AndIOp>(loc, tid, one);
    auto expect = builder.create<arith::ConstantIntOp>(loc, even ? 0 : 1, 32);
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, bit, expect);
}

Value lessThan(OpBuilder &builder, Location loc, Value tid, std::int64_t bound) {
    auto c = builder.create<arith::ConstantIntOp>(loc, bound, 32);
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, tid, c);
}

Value modEquals(OpBuilder &builder, Location loc, Value tid, std::int64_t mod, std::int64_t rem) {
    auto modC = builder.create<arith::ConstantIntOp>(loc, mod, 32);
    auto remC = builder.create<arith::ConstantIntOp>(loc, rem, 32);
    auto m = builder.create<arith::RemSIOp>(loc, tid, modC);
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, m, remC);
}

} // namespace patterns
} // namespace simt::fuzz
