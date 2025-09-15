use indexmap::IndexMap;
use serde::{Serialize, Deserialize};

pub type InstrId = u32;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Scope { Thread, Subgroup, Workgroup }
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SyncLevel { Independent, Synchronized }
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemSem { None, Acquire, Release, AcqRel }

#[derive(Debug, Clone)]
pub struct OperandDesc { pub name: String, pub ty: String }
#[derive(Debug, Clone)]
pub struct ResultDesc  { pub name: String, pub ty: String }

#[derive(Debug, Clone)]
pub struct InstructionSpec {
    pub name: String,
    pub operands: Vec<OperandDesc>,
    pub results: Vec<ResultDesc>,
    pub has_scope: bool,
    pub has_sync: bool,
    pub has_memsem: bool,
    pub needs_subgroup: bool,
    pub needs_shared_mem: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Params {
    pub scope: Option<Scope>,
    pub sync: Option<SyncLevel>,
    pub mem: Option<MemSem>,
    #[serde(flatten)]
    pub rest: IndexMap<String, serde_json::Value>,
}

pub trait InterpreterHandler: Send + Sync {
    fn interpret(&self, id: InstrId, params: &Params,
                 inputs: &[serde_json::Value],
                 outputs: &mut Vec<serde_json::Value>,
                 ctx: &mut crate::semantics::SemanticsContext) -> bool;
    fn expand_to_core(&self, _id: InstrId, _params: &Params) -> Option<()> { None }
}

pub struct Registry {
    specs: IndexMap<InstrId, InstructionSpec>,
    ids: IndexMap<String, InstrId>,
    next: InstrId,
    handlers: IndexMap<String, IndexMap<InstrId, Box<dyn InterpreterHandler>>>, // model -> (id->handler)
}

impl Registry {
    pub fn new() -> Self { Self { specs: IndexMap::new(), ids: IndexMap::new(), next: 1, handlers: IndexMap::new() } }
    pub fn register_instruction(&mut self, spec: InstructionSpec) -> InstrId {
        let id = self.next; self.next += 1;
        self.ids.insert(spec.name.clone(), id);
        self.specs.insert(id, spec); id
    }
    pub fn register_handler(&mut self, model: &str, id: InstrId, h: Box<dyn InterpreterHandler>) {
        self.handlers.entry(model.to_string()).or_default().insert(id, h);
    }
    pub fn lookup(&self, name: &str) -> Option<InstrId> { self.ids.get(name).copied() }
    pub fn handler(&self, model: &str, id: InstrId) -> Option<&(dyn InterpreterHandler)> {
        self.handlers.get(model)?.get(&id).map(|b| &**b)
    }
}
