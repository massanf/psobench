extern crate nalgebra as na;
use std::sync::Arc;

use nalgebra::DVector;

type OptimizationFunction = Arc<dyn Fn(&DVector<f64>) -> f64>;

#[derive(Clone)]
pub struct OptimizationProblem {
  #[allow(dead_code)]
  name: String,
  f: OptimizationFunction,
  domain: (f64, f64),
  dim: usize,
}

impl OptimizationProblem {
  pub fn new(name: String, f: OptimizationFunction, domain: (f64, f64), dim: usize) -> Self {
    Self { name, f, domain, dim }
  }

  #[allow(dead_code)]
  pub fn name(&self) -> &String {
    &self.name
  }
  pub fn f(&self, x: &DVector<f64>) -> f64 {
    (self.f)(x)
  }

  pub fn domain(&self) -> (f64, f64) {
    self.domain
  }

  pub fn dim(&self) -> usize {
    self.dim
  }
}
