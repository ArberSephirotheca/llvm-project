#pragma once

#include "simt-step/plugins/Registry.h"

#include <llvm/ADT/StringRef.h>

#include <memory>

namespace simt::plugins::examples {

std::shared_ptr<const InterpreterHandler> makeReduceAddHandler();
void registerReduceAdd(Registry &registry, llvm::StringRef model);

} // namespace simt::plugins::examples
