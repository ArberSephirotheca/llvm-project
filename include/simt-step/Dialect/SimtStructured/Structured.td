//===- Structured.td - Structured SIMT Dialect ------------*- tablegen -*-===//
//
// Explicit control-flow dialect modeling SIMT dynamic blocks and reconvergence.
//
//===----------------------------------------------------------------------===//

#ifndef SIMT_STRUCTURED_DIALECT_TD
#define SIMT_STRUCTURED_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

// Type constraints -----------------------------------------------------------------

def MaskLikeType : TypeConstraint<
    CPred<"::simt::structured::isMaskLikeType($_self)">,
    "mask-like type (simt_step.mask or signless integer)">;

// Dialect -------------------------------------------------------------------

def SimtStruct_Dialect : Dialect {
  let name = "simt_struct";
  let cppNamespace = "simt::structured";
  let summary = "Structured SIMT control-flow dialect";
  let description = [{
    Represents SIMT control flow using explicit block entities, mask management,
    and reconvergence hints. Lowering target for the higher-level simt_step
    dialect prior to interpretation or backend emission.
  }];
  let usePropertiesForAttributes = 0;
}

// Reconvergence policy enum -------------------------------------------------

def SimtStruct_ReconvergencePolicy : I32EnumAttr<"ReconvergencePolicy",
    "SIMT reconvergence policy",
    [I32EnumAttrCase<"PostDominator", 0>,
     I32EnumAttrCase<"Explicit", 1>,
     I32EnumAttrCase<"None", 2>]> {
  let cppNamespace = "::simt::structured";
}
def SimtStruct_ReconvergencePolicyAttr
    : EnumAttr<SimtStruct_Dialect, SimtStruct_ReconvergencePolicy,
               "reconvergence">;

// Block op ------------------------------------------------------------------

def SimtStruct_BlockOp : Op<SimtStruct_Dialect, "block",
    [IsolatedFromAbove, RegionBranchOpInterface]> {
  let summary = "Structured SIMT block";
  let description = [{
    Represents a basic block within the structured SIMT region. Each block has
    a symbol name for branches to reference and holds exactly one region with a
    single block of operations terminated by a structured terminator.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       OptionalAttr<FlatSymbolRefAttr>:$merge_target,
                       OptionalAttr<FlatSymbolRefAttr>:$continue_target,
                       OptionalAttr<SimtStruct_ReconvergencePolicyAttr>:$policy);
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    void getEntrySuccessorRegions(::llvm::ArrayRef<::mlir::Attribute> operands,
                                  ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
    void getSuccessorRegions(::mlir::RegionBranchPoint point,
                             ::llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions);
    bool areTypesCompatible(::mlir::Type lhs, ::mlir::Type rhs);
  }];

}

// Terminators ---------------------------------------------------------------

def SimtStruct_BranchOp : Op<SimtStruct_Dialect, "branch",
    [Terminator, HasParent<"BlockOp">]> {
  let summary = "Unconditional branch with mask";
  let arguments = (ins MaskLikeType:$mask,
                       FlatSymbolRefAttr:$target,
                       Variadic<AnyType>:$dest_operands);
}

def SimtStruct_CondBranchOp : Op<SimtStruct_Dialect, "cond_branch",
    [Terminator, HasParent<"BlockOp">, AttrSizedOperandSegments]> {
  let summary = "Conditional branch with explicit masks";
  let arguments = (ins I1:$condition,
                       MaskLikeType:$true_mask,
                       MaskLikeType:$false_mask,
                       FlatSymbolRefAttr:$true_target,
                       FlatSymbolRefAttr:$false_target,
                       Variadic<AnyType>:$true_operands,
                       Variadic<AnyType>:$false_operands,
                       OptionalAttr<FlatSymbolRefAttr>:$merge_target,
                       OptionalAttr<SimtStruct_ReconvergencePolicyAttr>:$policy);

  let hasVerifier = 1;
}

def SimtStruct_ReturnOp : Op<SimtStruct_Dialect, "return",
    [Terminator, HasParent<"BlockOp">]> {
  let summary = "Return from structured SIMT region";
  let arguments = (ins Variadic<AnyType>:$operands);
}

// Mask ops ------------------------------------------------------------------

def SimtStruct_MaskPushOp : Op<SimtStruct_Dialect, "mask_push",
    [HasParent<"BlockOp">]> {
  let summary = "Push a mask and continuation target";
  let arguments = (ins MaskLikeType:$mask,
                       OptionalAttr<FlatSymbolRefAttr>:$merge_target,
    OptionalAttr<FlatSymbolRefAttr>:$continue_target,
    OptionalAttr<I64Attr>:$loop_mask);
}

def SimtStruct_MaskPopOp : Op<SimtStruct_Dialect, "mask_pop",
    [HasParent<"BlockOp">]> {
  let summary = "Pop the next mask from the merge stack";
  let results = (outs MaskLikeType:$mask);
}

def SimtStruct_MaskMergeOp : Op<SimtStruct_Dialect, "mask_merge",
    [HasParent<"BlockOp">]> {
  let summary = "Merge an incoming mask with the active mask";
  let arguments = (ins MaskLikeType:$incoming);
  let results = (outs MaskLikeType:$merged);

  let hasVerifier = 1;
}

#endif // SIMT_STRUCTURED_DIALECT_TD
