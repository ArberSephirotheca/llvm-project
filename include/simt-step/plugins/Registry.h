#pragma once

#include "simt-step/core/Dialect.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/JSON.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simt::semantics {
struct SemanticsContext;
}

namespace simt::plugins {

using InstructionId = std::uint32_t;

enum class SyncLevel {
    Independent,
    Synchronized,
};

struct OperandDesc {
    std::string name;
    std::string type;
};

struct ResultDesc {
    std::string name;
    std::string type;
};

struct InstructionSpec {
    std::string name;
    std::vector<OperandDesc> operands;
    std::vector<ResultDesc> results;
    bool hasScope = false;
    bool hasSync = false;
    bool hasMemSem = false;
    bool needsSubgroup = false;
    bool needsSharedMemory = false;
};

struct Params {
    std::optional<core::Scope> scope;
    std::optional<SyncLevel> sync;
    std::optional<core::MemorySemantics> memory;
    llvm::json::Object rest;
};

class InterpreterHandler {
public:
    virtual ~InterpreterHandler() = default;

    virtual bool interpret(InstructionId id,
                           const Params &params,
                           llvm::ArrayRef<llvm::json::Value> inputs,
                           std::vector<llvm::json::Value> &outputs,
                           semantics::SemanticsContext &context) const = 0;
};

class Registry {
public:
    Registry();

    InstructionId registerInstruction(InstructionSpec spec);
    void registerHandler(llvm::StringRef model,
                         InstructionId id,
                         std::shared_ptr<const InterpreterHandler> handler);

    std::optional<InstructionId> lookup(llvm::StringRef name) const;
    const InterpreterHandler *getHandler(llvm::StringRef model, InstructionId id) const;

private:
    llvm::DenseMap<InstructionId, InstructionSpec> specs_;
    llvm::StringMap<InstructionId> ids_;
    llvm::StringMap<llvm::DenseMap<InstructionId, std::shared_ptr<const InterpreterHandler>>> handlers_;
    InstructionId nextId_ = 1;
};

llvm::Expected<Params> parseParams(const llvm::json::Value &value);

} // namespace simt::plugins
