extern crate nalgebra as na;
use crate::functions;
use std::collections::hash_map::DefaultHasher;
use std::sync::Arc;
use std::{
  collections::HashMap,
  hash::{Hash, Hasher},
};

use nalgebra::DVector;

type OptimizationFunction = Arc<dyn Fn(&DVector<f64>) -> f64 + Sync + Send>;

#[derive(Debug, Clone)]
struct HashableDVectorF64ForMemo(DVector<f64>);

impl HashableDVectorF64ForMemo {
  fn calculate_hash(&self) -> u64 {
    let mut hasher = DefaultHasher::new();
    for elem in self.0.iter() {
      elem.to_bits().hash(&mut hasher);
    }
    hasher.finish()
  }
}

impl Hash for HashableDVectorF64ForMemo {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.calculate_hash().hash(state);
  }
}

impl PartialEq for HashableDVectorF64ForMemo {
  fn eq(&self, other: &Self) -> bool {
    self.0 == other.0
  }
}

impl Eq for HashableDVectorF64ForMemo {}

#[derive(Clone)]
pub struct Problem {
  #[allow(dead_code)]
  name: String,
  f: OptimizationFunction,
  domain: (f64, f64),
  dim: usize,
  cnt: usize,
  memo: HashMap<HashableDVectorF64ForMemo, f64>,
}

impl Problem {
  pub fn new(name: String, f: OptimizationFunction, domain: (f64, f64), dim: usize) -> Self {
    Self {
      name,
      f,
      domain,
      dim,
      cnt: 0,
      memo: HashMap::new(),
    }
  }

  #[allow(dead_code)]
  pub fn name(&self) -> &String {
    &self.name
  }

  pub fn f(&mut self, x: &DVector<f64>) -> f64 {
    let hash = &HashableDVectorF64ForMemo(x.clone());
    if self.memo.contains_key(hash) {
      return self.memo[hash];
    }
    let ans = (self.f)(x);
    self.cnt += 1;
    self.memo.insert(hash.clone(), ans);
    ans
  }

  pub fn f_no_memo(&mut self, x: &DVector<f64>) -> f64 {
    (self.f)(x)
  }

  pub fn clear_memo(&mut self) {
    self.memo.clear();
  }

  pub fn domain(&self) -> (f64, f64) {
    self.domain
  }

  pub fn dim(&self) -> usize {
    self.dim
  }

  pub fn cnt(&self) -> usize {
    self.cnt
  }
}

impl Default for Problem {
  fn default() -> Problem {
    Problem::new("DEFAULT".to_owned(), Arc::new(functions::sphere), (-100., 100.), 0)
  }
}
