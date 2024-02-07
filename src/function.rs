extern crate nalgebra as na;

use nalgebra::DVector;

type OptimizationFunction = fn(&DVector<f64>) -> f64;

#[derive(Clone)]
pub struct OptimizationProblem {
  name: String,
  f: OptimizationFunction,
  domain: (f64, f64),
}

impl OptimizationProblem {
  pub fn new(name: &str, f: OptimizationFunction, domain: (f64, f64)) -> Self {
    Self {
      name: name.to_owned(),
      f,
      domain,
    }
  }

  pub fn name(&self) -> &String {
    &self.name
  }
  pub fn f(&self, x: &DVector<f64>) -> f64 {
    (self.f)(x)
  }

  pub fn domain(&self) -> (f64, f64) {
    self.domain
  }
}
