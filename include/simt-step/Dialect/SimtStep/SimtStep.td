//===- SimtStep.td - SIMT-Step Dialect  -------------------------*- tablegen -*-===//
//
// Defines the SIMT-Step dialect shell and a minimal set of ops. This is a
// placeholder to bootstrap the TableGen-based workflow; the op set will be
// expanded to match the design doc.
//
//===----------------------------------------------------------------------===//

#ifndef SIMT_STEP_DIALECT_TD
#define SIMT_STEP_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

// Dialect definition -------------------------------------------------------
def SimtStep_Dialect : Dialect {
  let name = "simt_step";
  let cppNamespace = "simt::dialect";
  let summary = "SIMT semantics dialect";
  let description = [{
    Dialect capturing warp-level operations, synchronization, and plugin
    extension points for SIMT semantics experiments.
  }];
  let usePropertiesForAttributes = 0;
  let useDefaultTypePrinterParser = 1;
}

// Trait markers -------------------------------------------------------------
def SimtStep_IndependentTrait : NativeOpTrait<"SimtIndependent">;
def SimtStep_SynchronizedTrait : NativeOpTrait<"SimtSynchronized">;
def SimtStep_CollectiveTrait : NativeOpTrait<"SimtCollective">;

// Enum attributes -----------------------------------------------------------

def SimtStep_Scope : I32EnumAttr<"Scope", "SIMT scope",
    [I32EnumAttrCase<"Thread", 0>,
     I32EnumAttrCase<"Subgroup", 1>,
     I32EnumAttrCase<"Workgroup", 2>]> {
  let cppNamespace = "::simt::dialect";
}
def SimtStep_ScopeEnumAttr : EnumAttr<SimtStep_Dialect, SimtStep_Scope, "scope">;

def SimtStep_MemorySemantics : I32EnumAttr<"MemorySemantics",
    "SIMT memory semantics",
    [I32EnumAttrCase<"None", 0>,
     I32EnumAttrCase<"Acquire", 1>,
     I32EnumAttrCase<"Release", 2>,
     I32EnumAttrCase<"AcqRel", 3>]> {
  let cppNamespace = "::simt::dialect";
}
def SimtStep_MemorySemanticsEnumAttr
    : EnumAttr<SimtStep_Dialect, SimtStep_MemorySemantics, "memsem">;

def SimtStep_MemorySpace : I32EnumAttr<"MemorySpace",
    "SIMT memory space",
    [I32EnumAttrCase<"Generic", 0>,
     I32EnumAttrCase<"Global", 1>,
     I32EnumAttrCase<"Shared", 2>,
     I32EnumAttrCase<"Private", 3>]> {
  let cppNamespace = "::simt::dialect";
}
def SimtStep_MemorySpaceEnumAttr
    : EnumAttr<SimtStep_Dialect, SimtStep_MemorySpace, "memspace">;

// Types --------------------------------------------------------------------

def SimtStep_MaskType : TypeDef<SimtStep_Dialect, "Mask"> {
  let mnemonic = "mask";
  let summary = "Bitmask type representing active lanes";
  let description = [{
    Represents a lane activity mask. The bit width defaults to 64 but may be
    customized via the `width` parameter.
  }];

  let parameters = (ins DefaultValuedParameter<"uint64_t", "64">:$width);
  let assemblyFormat = [{ (`<` $width^ `>`)? }];
  let genVerifyDecl = 1;

}

def SimtStep_ResourceType : TypeDef<SimtStep_Dialect, "Resource"> {
  let mnemonic = "resource";
  let summary = "SIMT buffer resource handle";
  let description = [{
    Represents a handle to a GPU buffer resource. The type records the memory
    space (e.g., global, shared) and the element type stored in the buffer.
  }];

  let parameters = (ins EnumParameter<SimtStep_MemorySpace>:$memorySpace,
                        TypeParameter<"mlir::Type", "element type">:$elementType);

  let assemblyFormat = [{
        `<` $memorySpace `,` $elementType `>`
  }];
  let genVerifyDecl = 1;
}

// Minimal op examples ------------------------------------------------------

def SimtStep_WaveAllOp : Op<SimtStep_Dialect, "wave_all"> {
  let summary = "Wave-level logical AND over lane predicates";
  let description = [{
    Computes whether all active lanes evaluate the predicate to true. The final
    operation set will include operands for mask/scope. This stub takes a single
    `i1` per-lane predicate and returns `i1`.
  }];

  let arguments = (ins I1:$predicate);
  let results = (outs I1:$result);

  let assemblyFormat = "`(` $predicate `)` attr-dict";
}

def SimtStep_WaveAnyOp : Op<SimtStep_Dialect, "wave_any"> {
  let summary = "Wave-level logical OR over lane predicates";
  let description = [{
    Computes whether any active lane evaluates the predicate to true. Returns a
    single `i1` result representing the wave-wide aggregate.
  }];

  let arguments = (ins I1:$predicate);
  let results = (outs I1:$result);

  let assemblyFormat = "`(` $predicate `)` attr-dict";
}

def SimtStep_WaveBallotOp : Op<SimtStep_Dialect, "wave_ballot"> {
  let summary = "Wave-level ballot producing a bitmask of lanes with predicate true";
  let description = [{
    Produces an integer mask (typically 64-bit) encoding which active lanes have
    the predicate set. The operand is an `i1` per-lane predicate, the result is
    an `i64` mask.
  }];

  let arguments = (ins I1:$predicate);
  let results = (outs I64:$mask);

  let assemblyFormat = "`(` $predicate `)` attr-dict";
}

def SimtStep_MaskPushOp : Op<SimtStep_Dialect, "mask_push"> {
  let summary = "Push an explicit mask onto the execution stack";
  let description = [{
    Pushes a new active mask. Placeholder for the structured lowering pipeline;
    not expected to appear in source-level IR.
  }];

  let arguments = (ins I64:$mask);

  let assemblyFormat = "$mask attr-dict";
}

def SimtStep_MaskPopOp : Op<SimtStep_Dialect, "mask_pop"> {
  let summary = "Pop the currently active mask";
  let description = [{
    Restores the previous mask. Placeholder awaiting structured dialect and
    interpreter lowering.
  }];

  let assemblyFormat = "attr-dict";
}

def SimtStep_MaskMergeOp : Op<SimtStep_Dialect, "mask_merge"> {
  let summary = "Merge an incoming mask with the current active mask";
  let description = [{
    Combines control-flow masks. Returns the merged mask for downstream use.
  }];

  let arguments = (ins I64:$incoming);
  let results = (outs I64:$merged);

  let assemblyFormat = "$incoming attr-dict";
}

def SimtStep_BufferLoadOp : Op<SimtStep_Dialect, "buffer.load",
    [MemoryEffects<[MemRead]>]> {
  let summary = "Load an element from a buffer resource";
  let description = [{
    Reads the element at the given index from the buffer resource operand and
    produces it as the result. The result type must match the resource element
    type.
  }];

  let arguments = (ins SimtStep_ResourceType:$resource,
                       AnyInteger:$index);
  let results = (outs AnyType:$result);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$resource, "mlir::Value":$index)>
  ];
}

def SimtStep_BufferStoreOp : Op<SimtStep_Dialect, "buffer.store",
    [MemoryEffects<[MemWrite]>]> {
  let summary = "Store an element into a buffer resource";
  let description = [{
    Writes the provided value into the buffer resource at the given index. The
    value type must match the resource element type.
  }];

  let arguments = (ins SimtStep_ResourceType:$resource,
                       AnyInteger:$index,
                       AnyType:$value);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$resource, "mlir::Value":$index,
                    "mlir::Value":$value)>
  ];
}

def SimtStep_BufferAtomicAddOp : Op<SimtStep_Dialect, "buffer.atomic_add",
    [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Atomic add on a buffer element";
  let description = [{
    Performs an atomic add on the buffer resource at the given index. Returns
    the original value prior to the addition.
  }];

  let arguments = (ins SimtStep_ResourceType:$resource,
                       AnyInteger:$index,
                       AnyType:$value);
  let results = (outs AnyType:$oldValue);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$resource, "mlir::Value":$index,
                    "mlir::Value":$value)>
  ];
}

def SimtStep_BufferAtomicExchangeOp : Op<SimtStep_Dialect,
    "buffer.atomic_exchange", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Atomic exchange on a buffer element";
  let description = [{
    Performs an atomic exchange, writing the provided value and returning the
    value previously stored at the buffer index.
  }];

  let arguments = (ins SimtStep_ResourceType:$resource,
                       AnyInteger:$index,
                       AnyType:$value);
  let results = (outs AnyType:$oldValue);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$resource, "mlir::Value":$index,
                    "mlir::Value":$value)>
  ];
}

def SimtStep_BufferAtomicCompareExchangeOp : Op<SimtStep_Dialect,
    "buffer.atomic_compare_exchange", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Atomic compare-and-exchange on a buffer element";
  let description = [{
    Performs an atomic compare-and-exchange. If the current element value
    matches `compare`, it is replaced with `value`. The previous element value
    is returned regardless of whether the swap occurred.
  }];

  let arguments = (ins SimtStep_ResourceType:$resource,
                       AnyInteger:$index,
                       AnyType:$compare,
                       AnyType:$value);
  let results = (outs AnyType:$oldValue);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$resource, "mlir::Value":$index,
                    "mlir::Value":$compare, "mlir::Value":$value)>
  ];
}

def SimtStep_BufferAtomicMinOp : Op<SimtStep_Dialect,
    "buffer.atomic_min", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Atomic minimum on a buffer element";
  let description = [{
    Replaces the buffer element with the minimum of the current value and the
    operand, returning the original value.
  }];

  let arguments = (ins SimtStep_ResourceType:$resource,
                       AnyInteger:$index,
                       AnyType:$value);
  let results = (outs AnyType:$oldValue);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$resource, "mlir::Value":$index,
                    "mlir::Value":$value)>
  ];
}

def SimtStep_BufferAtomicMaxOp : Op<SimtStep_Dialect,
    "buffer.atomic_max", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Atomic maximum on a buffer element";
  let description = [{
    Replaces the buffer element with the maximum of the current value and the
    operand, returning the original value.
  }];

  let arguments = (ins SimtStep_ResourceType:$resource,
                       AnyInteger:$index,
                       AnyType:$value);
  let results = (outs AnyType:$oldValue);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$resource, "mlir::Value":$index,
                    "mlir::Value":$value)>
  ];
}

def SimtStep_BufferAtomicAndOp : Op<SimtStep_Dialect,
    "buffer.atomic_and", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Atomic bitwise AND on a buffer element";
  let description = [{
    Applies a bitwise AND with the operand and the buffer element. Returns the
    previous element value.
  }];

  let arguments = (ins SimtStep_ResourceType:$resource,
                       AnyInteger:$index,
                       AnyType:$value);
  let results = (outs AnyType:$oldValue);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$resource, "mlir::Value":$index,
                    "mlir::Value":$value)>
  ];
}

def SimtStep_BufferAtomicOrOp : Op<SimtStep_Dialect, "buffer.atomic_or",
    [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Atomic bitwise OR on a buffer element";
  let description = [{
    Applies a bitwise OR with the operand and the buffer element. Returns the
    previous element value.
  }];

  let arguments = (ins SimtStep_ResourceType:$resource,
                       AnyInteger:$index,
                       AnyType:$value);
  let results = (outs AnyType:$oldValue);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$resource, "mlir::Value":$index,
                    "mlir::Value":$value)>
  ];
}

def SimtStep_BufferAtomicXorOp : Op<SimtStep_Dialect, "buffer.atomic_xor",
    [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Atomic bitwise XOR on a buffer element";
  let description = [{
    Applies a bitwise XOR with the operand and the buffer element. Returns the
    previous element value.
  }];

  let arguments = (ins SimtStep_ResourceType:$resource,
                       AnyInteger:$index,
                       AnyType:$value);
  let results = (outs AnyType:$oldValue);

  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$resource, "mlir::Value":$index,
                    "mlir::Value":$value)>
  ];
}

// Control flow --------------------------------------------------------------

def SimtStep_SwitchOp : Op<SimtStep_Dialect, "switch", [NoRegionArguments]> {
  let summary = "Structured SIMT switch";
  let description = [{
    Structured container for lowering switch statements. The selector operand
    chooses among cases enumerated in the `case_values` attribute. Each case
    and the default branch end with `simt_step.yield`, forwarding the
    switch-carried values.
  }];

  let arguments = (ins AnyType:$selector,
                       Variadic<AnyType>:$initialValues,
                       DenseI64ArrayAttr:$case_values);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$caseBody);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::Value":$selector,
                    "mlir::ValueRange":$initialValues,
                    "mlir::ArrayRef<int64_t>":$caseValues)>
  ];

  let hasVerifier = 1;
}

def SimtStep_YieldOp : Op<SimtStep_Dialect, "yield",
    [Terminator, ParentOneOf<["IfOp", "LoopOp", "SwitchOp"]>]> {
  let summary = "Yield from a structured SIMT region";
  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

def SimtStep_IfOp : Op<SimtStep_Dialect, "if",
    [SingleBlockImplicitTerminator<"YieldOp">,
     NoRegionArguments]> {
  let summary = "SIMT if-then-else";
  let description = [{
    Control flow construct with optional else region producing SSA results.
    Each region terminates with `simt_step.yield` to forward per-lane values.
  }];

  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                 MaxSizedRegion<1>:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::Value":$condition,
                    "bool":$withElseRegion)>,
    OpBuilder<(ins "mlir::Value":$condition, "bool":$withElseRegion)>
  ];

  let hasVerifier = 1;
}

def SimtStep_ConditionOp : Op<SimtStep_Dialect, "condition",
    [HasParent<"LoopOp">, Terminator]> {
  let summary = "Loop continuation condition";
  let description = [{
    Terminator of the loop pre-header region. The first operand specifies
    whether execution proceeds into the loop body. The remaining operands are
    forwarded either to the body region or to the results of the loop.
  }];

  let arguments = (ins I1:$condition, Variadic<AnyType>:$forwarded);

  let assemblyFormat =
      [{ `(` $condition `)` attr-dict ($forwarded^ `:` type($forwarded))? }];
}

def SimtStep_LoopOp : Op<SimtStep_Dialect, "loop",
    [SingleBlock, NoRegionArguments]> {
  let summary = "SIMT structured loop";
  let description = [{
    Structured loop op with separate setup and body regions. The "prepare"
    region evaluates the continuation condition and can forward values to the
    body or loop results. The body region yields updated loop-carried values
    back to the prepare region.
  }];

  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$prepareRegion,
                 SizedRegion<1>:$bodyRegion);

}

def SimtStep_BreakOp : Op<SimtStep_Dialect, "break",
    [Terminator, HasParent<"LoopOp">]> {
  let summary = "Structured break from simt.loop";
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = [{ attr-dict ($results^ `:` type($results))? }];
}

def SimtStep_ContinueOp : Op<SimtStep_Dialect, "continue",
    [Terminator, HasParent<"LoopOp">]> {
  let summary = "Structured continue within simt.loop";
  let arguments = (ins Variadic<AnyType>:$results);
  let assemblyFormat = [{ attr-dict ($results^ `:` type($results))? }];
}

def SimtStep_BarrierOp : Op<SimtStep_Dialect, "barrier"> {
  let summary = "Synchronize lanes at a given scope";
  let description = [{
    Synchronize active lanes according to the provided `scope` and `memsem`
    attributes. The scope defaults to workgroup when omitted.
  }];

  let arguments = (ins OptionalAttr<SimtStep_ScopeEnumAttr>:$scope,
                       OptionalAttr<SimtStep_MemorySemanticsEnumAttr>:$memsem);

  let hasCustomAssemblyFormat = 1;
}

def SimtStep_FenceOp : Op<SimtStep_Dialect, "fence"> {
  let summary = "Memory fence at a given scope";
  let description = [{
    Orders memory accesses at the given `scope`, applying the specified
    semantics and optional `memspace` target.
  }];

  let arguments = (ins OptionalAttr<SimtStep_ScopeEnumAttr>:$scope,
                       OptionalAttr<SimtStep_MemorySemanticsEnumAttr>:$memsem,
                       OptionalAttr<SimtStep_MemorySpaceEnumAttr>:$memspace);

  let hasCustomAssemblyFormat = 1;
}

def SimtStep_LaneIdOp : Op<SimtStep_Dialect, "lane_id"> {
  let summary = "Query the current lane identifier";
  let description = [{
    Returns the numeric lane id within the subgroup.
  }];

  let results = (outs Index:$lane);

  let assemblyFormat = "attr-dict";
}

def SimtStep_ActiveMaskOp : Op<SimtStep_Dialect, "active_mask"> {
  let summary = "Expose the current active lane mask";
  let description = [{
    Returns an `i64` mask representing currently active lanes. Planned future
    work will make the bit-width configurable via attributes.
  }];

  let results = (outs I64:$mask);
}

def SimtStep_MemLoadOp : Op<SimtStep_Dialect, "mem_load"> {
  let summary = "Load a value from memory";
  let description = [{
    Placeholder memory load op. Takes an address operand and yields a value.
    Actual typing rules will be refined once the lowering pipeline is defined.
  }];

  let arguments = (ins AnyType:$address);
  let results = (outs AnyType:$value);
}

def SimtStep_MemStoreOp : Op<SimtStep_Dialect, "mem_store"> {
  let summary = "Store a value to memory";
  let description = [{
    Placeholder memory store op writing the supplied value to the given address.
  }];

  let arguments = (ins AnyType:$value, AnyType:$address);
}

def SimtStep_CustomOp : Op<SimtStep_Dialect, "custom"> {
  let summary = "Extension hook for plugin-defined instructions";
  let description = [{
    Represents a plugin-provided instruction. Carries an `instr` string attribute
    naming the instruction plus an optional `params` dictionary for structured
    metadata.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$outputs);

  let hasVerifier = 1;
}

#endif // SIMT_STEP_DIALECT_TD
