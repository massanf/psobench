use crate::optimization_problem;
use nalgebra::DVector;
use optimization_problem::Problem;
use rand::distributions::{Distribution, Uniform};
use std::fs;
use std::io;
use std::path::PathBuf;

pub fn uniform_distribution(low: &DVector<f64>, high: &DVector<f64>) -> DVector<f64> {
  let mut rng = rand::thread_rng();
  DVector::from_iterator(
    low.len(),
    (0..low.len()).map(|i| Uniform::new(low[i], high[i]).sample(&mut rng)),
  )
}

// TODO: There must be a better place to put this.
pub fn random_init_pos(problem: &Problem) -> DVector<f64> {
  let b_lo: DVector<f64> = DVector::from_element(problem.dim(), problem.domain().0);
  let b_up: DVector<f64> = DVector::from_element(problem.dim(), problem.domain().1);
  uniform_distribution(&b_lo, &b_up)
}

// TODO: There must be a better place to put this.
pub fn random_init_vel(problem: &Problem) -> DVector<f64> {
  let b_lo: DVector<f64> = DVector::from_element(problem.dim(), problem.domain().0);
  let b_up: DVector<f64> = DVector::from_element(problem.dim(), problem.domain().1);

  uniform_distribution(
    &DVector::from_iterator(problem.dim(), (&b_up - &b_lo).iter().map(|b| -b.abs())),
    &DVector::from_iterator(problem.dim(), (&b_up - &b_lo).iter().map(|b| b.abs())),
  )
}

pub fn create_directory(path: PathBuf, clear: bool) {
  // Handle output directory creation / deletion
  if path.exists() {
    if !clear {
      return;
    }
    println!("The directory {:?} already exists. Overwrite? (y/n)", path);
    let mut user_input = String::new();
    let _ = io::stdin().read_line(&mut user_input);

    match user_input.trim().to_lowercase().as_str() {
      "y" => {
        let _ = fs::remove_dir_all(path.clone());
      }
      _ => {
        println!("Cancelled.");
        std::process::exit(1);
      }
    }
  }
  let _ = fs::create_dir_all(path);
}
