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

// Dialect definition -------------------------------------------------------
def SimtStep_Dialect : Dialect {
  let name = "simt_step";
  let cppNamespace = "simt::dialect";
  let summary = "SIMT semantics dialect";
  let description = [{
    Dialect capturing warp-level operations, synchronization, and plugin
    extension points for SIMT semantics experiments.
  }];
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

def SimtStep_BarrierOp : Op<SimtStep_Dialect, "barrier"> {
  let summary = "Synchronize lanes at a given scope";
  let description = [{
    Placeholder barrier op; real version will use attributes for scope and
    memory semantics.
  }];

  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def SimtStep_FenceOp : Op<SimtStep_Dialect, "fence"> {
  let summary = "Memory fence at a given scope";
  let description = [{
    Ensures ordering of memory operations according to the specified semantics.
    Currently modeled as a placeholder with no explicit attributes.
  }];

  let assemblyFormat = "attr-dict";
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

  let assemblyFormat = "attr-dict";
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
