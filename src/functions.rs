use nalgebra::DVector;

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
pub fn skewed_sphere(v: &DVector<f64>) -> f64 {
  let mut sum = 0.0;
  for i in 0..v.len() {
    sum += (v[i] - 0.8) * (v[i] - 0.8);
  }
  sum
}

#[allow(dead_code)]
pub fn rosenbrock(v: &DVector<f64>) -> f64 {
  let mut sum = 0.0;
  for i in 0..v.len() - 1 {
    let x = v[i];
    let x_next = v[i + 1];
    sum += 100.0 * (x_next - x * x).powi(2) + (1.0 - x).powi(2);
  }
  sum
}

#[allow(dead_code)]
pub fn rastrigin(v: &DVector<f64>) -> f64 {
  let a = 10.0;
  let mut sum = a * v.len() as f64;
  for i in 0..v.len() {
    sum += v[i] * v[i] - a * (2.0 * std::f64::consts::PI * v[i]).cos();
  }
  sum
}

#[allow(dead_code)]
pub fn cec17_impl(v: &DVector<f64>, func_num: usize) -> f64 {
  let x = v.clone();
  let nx = v.len() as usize;
  let mut result = 0.0f64;
  unsafe {
    cec17_test_func(x.as_ptr(), &mut result, nx, 1, func_num);
  }
  result
}
