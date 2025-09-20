#include "simt-step/plugins/Registry.h"

#include "simt-step/semantics/SemanticsContext.h"

#include <llvm/Support/Error.h>

namespace simt::plugins {

using namespace llvm;

Registry::Registry() = default;

InstructionId Registry::registerInstruction(InstructionSpec spec) {
    const InstructionId id = nextId_++;
    ids_.insert_or_assign(spec.name, id);
    specs_.insert_or_assign(id, std::move(spec));
    return id;
}

void Registry::registerHandler(StringRef model,
                               InstructionId id,
                               std::shared_ptr<const InterpreterHandler> handler) {
    handlers_[model].insert_or_assign(id, std::move(handler));
}

std::optional<InstructionId> Registry::lookup(StringRef name) const {
    if (auto it = ids_.find(name); it != ids_.end()) {
        return it->second;
    }
    return std::nullopt;
}

const InterpreterHandler *Registry::getHandler(StringRef model, InstructionId id) const {
    if (auto modelIt = handlers_.find(model); modelIt != handlers_.end()) {
        if (auto handlerIt = modelIt->second.find(id); handlerIt != modelIt->second.end()) {
            return handlerIt->second.get();
        }
    }
    return nullptr;
}

static Expected<core::Scope> parseScope(StringRef value) {
    if (value.equals_insensitive("thread")) {
        return core::Scope::Thread;
    }
    if (value.equals_insensitive("subgroup")) {
        return core::Scope::Subgroup;
    }
    if (value.equals_insensitive("workgroup")) {
        return core::Scope::Workgroup;
    }
    return make_error<StringError>("unknown scope value", inconvertibleErrorCode());
}

static Expected<SyncLevel> parseSyncLevel(StringRef value) {
    if (value.equals_insensitive("independent")) {
        return SyncLevel::Independent;
    }
    if (value.equals_insensitive("synchronized")) {
        return SyncLevel::Synchronized;
    }
    return make_error<StringError>("unknown sync level", inconvertibleErrorCode());
}

static Expected<core::MemorySemantics> parseMemSem(StringRef value) {
    if (value.equals_insensitive("none")) {
        return core::MemorySemantics::None;
    }
    if (value.equals_insensitive("acquire")) {
        return core::MemorySemantics::Acquire;
    }
    if (value.equals_insensitive("release")) {
        return core::MemorySemantics::Release;
    }
    if (value.equals_insensitive("acqrel") || value.equals_insensitive("acquire_release")) {
        return core::MemorySemantics::AcquireRelease;
    }
    return make_error<StringError>("unknown memory semantics", inconvertibleErrorCode());
}

Expected<Params> parseParams(const json::Value &value) {
    const auto *object = value.getAsObject();
    if (!object) {
        return make_error<StringError>("expected params object", inconvertibleErrorCode());
    }

    Params params;
    for (const auto &entry : *object) {
        const StringRef key = entry.first;
        if (key == "scope") {
            if (auto str = entry.second.getAsString()) {
                auto parsed = parseScope(*str);
                if (!parsed) {
                    return parsed.takeError();
                }
                params.scope = *parsed;
            }
            continue;
        }
        if (key == "sync") {
            if (auto str = entry.second.getAsString()) {
                auto parsed = parseSyncLevel(*str);
                if (!parsed) {
                    return parsed.takeError();
                }
                params.sync = *parsed;
            }
            continue;
        }
        if (key == "mem") {
            if (auto str = entry.second.getAsString()) {
                auto parsed = parseMemSem(*str);
                if (!parsed) {
                    return parsed.takeError();
                }
                params.memory = *parsed;
            }
            continue;
        }

        params.rest.try_emplace(key, entry.second);
    }

    return params;
}

} // namespace simt::plugins
