use crate::particle_trait::{Position, Velocity};
use crate::problem;
use crate::DataExporter;
use crate::ParamValue;
use crate::ParticleOptimizer;
use nalgebra::DVector;
use problem::Problem;
use rand::distributions::{Distribution, Uniform};
use rayon::prelude::*;
use std::collections::HashMap;
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
            let _ = fs::create_dir_all(path);
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
            let _ = fs::create_dir_all(path);
          }
          _ => {
            eprintln!("Cancelled.");
            std::process::exit(1);
          }
        }
      }
      (false, false) => {}
    },
  }
}

pub fn run_attempts<U: Position + Velocity, T: ParticleOptimizer<U> + DataExporter<U>>(
  params: HashMap<String, ParamValue>,
  name: String,
  problem: Problem,
  out_directory: PathBuf,
  iterations: usize,
  attempts: usize,
  save_data: bool,
  bar: &indicatif::ProgressBar,
) -> Result<(), Box<dyn std::error::Error>> {
  (0..attempts).into_par_iter().for_each(|attempt| {
    let mut pso: T = T::new(
      name.clone(),
      problem.clone(),
      params.clone(),
      out_directory.join(format!("{}", attempt)),
    );
    pso.run(iterations);
    let _ = pso.save_summary();
    let _ = pso.save_config(&params);
    if save_data {
      let _ = pso.save_data();
    }
    bar.inc(1);
  });
  Ok(())
}
