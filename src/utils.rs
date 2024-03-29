use crate::problem;
use nalgebra::DVector;
use problem::Problem;
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

pub fn create_directory(path: PathBuf, addable: bool, ask_clear: bool) {
  assert!(!(!addable && !ask_clear));
  match path.exists() {
    false => {
      let _ = fs::create_dir_all(path);
    }
    true => match (addable, ask_clear) {
      (true, true) => {
        println!(
          "The directory {:?} already exists. Clear? (Not clearing will add data). (y/n)",
          path
        );
        let mut user_input = String::new();
        let _ = io::stdin().read_line(&mut user_input);

        match user_input.trim().to_lowercase().as_str() {
          "y" => {
            let _ = fs::remove_dir_all(path.clone());
          }
          _ => {}
        }
      }
      (true, false) => {}
      (false, true) => {
        println!("The directory {:?} already exists. Clear? (y/n)", path);
        let mut user_input = String::new();
        let _ = io::stdin().read_line(&mut user_input);

        match user_input.trim().to_lowercase().as_str() {
          "y" => {
            let _ = fs::remove_dir_all(path.clone());
          }
          _ => {}
        }
      }
      (false, false) => {}
    },
  }
}
