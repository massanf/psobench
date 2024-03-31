extern crate nalgebra as na;
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
    if self.memo.contains_key(&HashableDVectorF64ForMemo(x.clone())) {
      return self.memo[&HashableDVectorF64ForMemo(x.clone())];
    }
    let ans = (self.f)(x);
    self.cnt += 1;
    self.memo.insert(HashableDVectorF64ForMemo(x.clone()), ans);
    ans
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
