use nalgebra::DVector;
use rand::distributions::{Distribution, Uniform};

pub fn uniform_distribution(low: &DVector<f64>, high: &DVector<f64>) -> DVector<f64> {
  let mut rng = rand::thread_rng();
  DVector::from_iterator(
    low.len(),
    (0..low.len()).map(|i| Uniform::new(low[i], high[i]).sample(&mut rng)),
  )
}

// TODO: There must be a better place to put this.
pub fn random_init_pos(dimensions: usize) -> DVector<f64> {
  let b_lo: DVector<f64> = DVector::from_element(dimensions, -1.0);
  let b_up: DVector<f64> = DVector::from_element(dimensions, 1.0);
  uniform_distribution(&b_lo, &b_up)
}

// TODO: There must be a better place to put this.
pub fn random_init_vel(dimensions: usize) -> DVector<f64> {
  let b_lo: DVector<f64> = DVector::from_element(dimensions, -1.0);
  let b_up: DVector<f64> = DVector::from_element(dimensions, 1.0);

  uniform_distribution(
    &DVector::from_iterator(dimensions, (&b_up - &b_lo).iter().map(|b| -b.abs())),
    &DVector::from_iterator(dimensions, (&b_up - &b_lo).iter().map(|b| b.abs())),
  )
}

pub fn format_dvector(vec: &DVector<f64>) -> String {
  let mut result = String::new();
  for i in 0..vec.len() {
    result.push_str(&format!("{:.3}", vec[i]));
    if i != vec.len() - 1 {
      result.push_str(", ");
    }
  }
  result
}
