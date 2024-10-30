use nalgebra::DVector;

extern "C" {
  fn cec17_test_func(x: *const f64, f: *mut f64, nx: usize, mx: usize, func_num: usize);
}

#[allow(dead_code)]
pub fn sphere(v: &DVector<f64>) -> f64 {
  let mut sum = 0.0;
  for i in 0..v.len() {
    sum += (v[i] + 50.) * (v[i] + 50.) * (v[i] + 50.) * (v[i] + 50.);
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
pub fn hyper_ellipsoid(v: &DVector<f64>) -> f64 {
  let mut sum = 0.0;
  for k in 0..v.len() {
    sum += (k as f64 * v[k]).powi(2);
  }
  sum
}

#[allow(dead_code)]
pub fn griewank(v: &DVector<f64>) -> f64 {
  let sum_part: f64 = v.iter().map(|&u| u.powi(2)).sum::<f64>() / 4000.0;
  let product_part: f64 = v.iter().enumerate().map(|(i, &u)| (u / ((i + 1) as f64).sqrt()).cos()).product();

  sum_part - product_part + 1.0
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
    sum += (v[i] - 50.) * (v[i] - 50.) - a * (2.0 * std::f64::consts::PI * (v[i] - 50.)).cos();
    // sum += (v[i].powi(2) - 50.0_f64.powi(2)).powi(2);
    // sum += (v[i] + 50.) * (v[i] + 50.);
  }
  sum
}

#[allow(dead_code)]
pub fn cec17_impl(v: &DVector<f64>, func_num: usize) -> f64 {
  let x = v.clone();
  let nx = v.len();
  let mut result = 0.0f64;
  unsafe {
    cec17_test_func(x.as_ptr(), &mut result, nx, 1, func_num);
  }
  // println!("result: {:?}", result);
  result
}
