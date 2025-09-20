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

def SimtStep_BarrierOp : Op<SimtStep_Dialect, "barrier"> {
  let summary = "Synchronize lanes at a given scope";
  let description = [{
    Placeholder barrier op; real version will use attributes for scope and
    memory semantics.
  }];

  let assemblyFormat = "attr-dict";
}

#endif // SIMT_STEP_DIALECT_TD
