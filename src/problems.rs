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

#[allow(dead_code)]
pub fn f1(dim: usize) -> Problem {
  Problem::new("Sphere".to_owned(), Arc::new(functions::sphere), (-1., 1.), dim)
}

#[allow(dead_code)]
pub fn f1_skewed(dim: usize) -> Problem {
  Problem::new(
    "Skewed Sphere".to_owned(),
    Arc::new(functions::skewed_sphere),
    (-1., 1.),
    dim,
  )
}

#[allow(dead_code)]
pub fn sphere_100(dim: usize) -> Problem {
  Problem::new(
    "Sphere100".to_owned(),
    Arc::new(functions::sphere),
    (-100., 100.),
    dim,
  )
}

#[allow(dead_code)]
pub fn rosenbrock_30(dim: usize) -> Problem {
  Problem::new(
    "Rosenbrock30".to_owned(),
    Arc::new(functions::rosenbrock),
    (-30., 30.),
    dim,
  )
}

#[allow(dead_code)]
pub fn griewank_600(dim: usize) -> Problem {
  Problem::new(
    "Griewank600".to_owned(),
    Arc::new(functions::griewank),
    (-600., 600.),
    dim,
  )
}

#[allow(dead_code)]
pub fn rastrigin_5_12(dim: usize) -> Problem {
  Problem::new(
    "Rastrigin5_12".to_owned(),
    Arc::new(functions::rastrigin),
    (-5.12, 5.12),
    dim,
  )
}

#[allow(dead_code)]
pub fn rastrigin_100(dim: usize) -> Problem {
  Problem::new(
    "Rastrigin100".to_owned(),
    Arc::new(functions::rastrigin),
    (-100., 100.),
    dim,
  )
}

#[allow(dead_code)]
pub fn hyper_ellipsoid_100(dim: usize) -> Problem {
  Problem::new(
    "HyperEllipsoid100".to_owned(),
    Arc::new(functions::hyper_ellipsoid),
    (-100., 100.),
    dim,
  )
}

#[allow(dead_code)]
pub fn f3(dim: usize) -> Problem {
  Problem::new(
    "Rastrigin".to_owned(),
    Arc::new(functions::rastrigin),
    (-5.12, 5.12),
    dim,
  )
}

#[allow(dead_code)]
pub fn cec17(func_num: usize, dim: usize) -> Problem {
  assert!(
    [2, 10, 20, 30, 50, 100].contains(&dim),
    "The dimensions for CEC2017 functions must be 2, 10, 20, 30, 50, or 100."
  );
  assert!(
    (1..=30).contains(&func_num),
    "CEC2017 contains 30 functions from F1 to F30 except for F2."
  );
  assert!(func_num != 2, "CEC2017 F2 has been deprecated.");
  Problem::new(
    format!("CEC2017_F{:02}", func_num),
    Arc::new(move |x: &DVector<f64>| functions::cec17_impl(x, func_num)),
    (-100., 100.),
    dim,
  )
}
