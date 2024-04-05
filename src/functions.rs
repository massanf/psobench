use crate::problem::Problem;
use nalgebra::DVector;
use std::sync::Arc;

extern "C" {
  fn cec17_test_func(x: *const f64, f: *mut f64, nx: usize, mx: usize, func_num: usize);
}

#[allow(dead_code)]
pub fn sphere(v: &DVector<f64>) -> f64 {
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
fn cec17_impl(v: &DVector<f64>, func_num: usize) -> f64 {
  let x = v.clone();
  let nx = v.len() as usize;
  let mut result = 0.0f64;
  unsafe {
    cec17_test_func(x.as_ptr(), &mut result, nx, 1, func_num);
  }
  result
}

#[allow(dead_code)]
pub fn f1(dim: usize) -> Problem {
  Problem::new("Sphere".to_owned(), Arc::new(sphere), (-1., 1.), dim)
}

#[allow(dead_code)]
pub fn f2(dim: usize) -> Problem {
  Problem::new("Rosenbrock".to_owned(), Arc::new(rosenbrock), (-30., 30.), dim)
}

#[allow(dead_code)]
pub fn f3(dim: usize) -> Problem {
  Problem::new("Rastrigin".to_owned(), Arc::new(rastrigin), (-5.12, 5.12), dim)
}

#[allow(dead_code)]
pub fn cec17(func_num: usize, dim: usize) -> Problem {
  assert!(
    vec![2, 10, 20, 30, 50, 100].contains(&dim),
    "The dimensions for CEC2017 functions must be 2, 10, 20, 30, 50, or 100."
  );
  assert!(
    func_num >= 1 && func_num <= 30,
    "CEC2017 contains 30 functions from F1 to F30 except for F2."
  );
  assert!(func_num != 2, "CEC2017 F2 has been deprecated.");
  Problem::new(
    format!("CEC2017_F{:02}", func_num),
    Arc::new(move |x: &DVector<f64>| cec17_impl(x, func_num)),
    (-100., 100.),
    dim,
  )
}
