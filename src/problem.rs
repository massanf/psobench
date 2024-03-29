extern crate nalgebra as na;
use std::sync::Arc;

use nalgebra::DVector;

type OptimizationFunction = Arc<dyn Fn(&DVector<f64>) -> f64 + Sync + Send>;

#[derive(Clone)]
pub struct Problem {
  #[allow(dead_code)]
  name: String,
  f: OptimizationFunction,
  domain: (f64, f64),
  dim: usize,
  cnt: usize,
}

impl Problem {
  pub fn new(name: String, f: OptimizationFunction, domain: (f64, f64), dim: usize) -> Self {
    Self {
      name,
      f,
      domain,
      dim,
      cnt: 0,
    }
  }

  #[allow(dead_code)]
  pub fn name(&self) -> &String {
    &self.name
  }

  pub fn f(&mut self, x: &DVector<f64>) -> f64 {
    let ans = (self.f)(x);
    self.cnt += 1;
    ans
  }

  pub fn domain(&self) -> (f64, f64) {
    self.domain
  }

  pub fn dim(&self) -> usize {
    self.dim
  }
}
