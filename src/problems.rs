use crate::function::OptimizationProblem;
use nalgebra::DVector;

#[allow(dead_code)]
fn sphere(v: &DVector<f64>) -> f64 {
  let mut sum = 0.0;
  for i in 0..v.len() {
    sum += v[i] * v[i];
  }
  sum
}

#[allow(dead_code)]
fn rosenbrock(v: &DVector<f64>) -> f64 {
  let mut sum = 0.0;
  for i in 0..v.len() - 1 {
    let x = v[i];
    let x_next = v[i + 1];
    sum += 100.0 * (x_next - x * x).powi(2) + (1.0 - x).powi(2);
  }
  sum
}

#[allow(dead_code)]
fn rastrigin(v: &DVector<f64>) -> f64 {
  let a = 10.0;
  let mut sum = a * v.len() as f64;
  for i in 0..v.len() {
    sum += v[i] * v[i] - a * (2.0 * std::f64::consts::PI * v[i]).cos();
  }
  sum
}

#[allow(dead_code)]
pub fn f1(dim: usize) -> OptimizationProblem {
  OptimizationProblem::new("Sphere", sphere, (-1., 1.), dim)
}

#[allow(dead_code)]
pub fn f2(dim: usize) -> OptimizationProblem {
  OptimizationProblem::new("Rosenbrock", rosenbrock, (-30., 30.), dim)
}

#[allow(dead_code)]
pub fn f3(dim: usize) -> OptimizationProblem {
  OptimizationProblem::new("Rastrigin", rastrigin, (-5.12, 5.12), dim)
}
