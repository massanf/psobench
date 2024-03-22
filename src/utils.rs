use crate::optimization_problem;
use crate::particle_trait::ParticleTrait;
use crate::pso_trait::PSOTrait;
use indicatif::ProgressBar;
use nalgebra::DVector;
use optimization_problem::OptimizationProblem;
use rand::distributions::{Distribution, Uniform};
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
pub fn random_init_pos(problem: &OptimizationProblem) -> DVector<f64> {
  let b_lo: DVector<f64> = DVector::from_element(problem.dim(), problem.domain().0);
  let b_up: DVector<f64> = DVector::from_element(problem.dim(), problem.domain().1);
  uniform_distribution(&b_lo, &b_up)
}

// TODO: There must be a better place to put this.
pub fn random_init_vel(problem: &OptimizationProblem) -> DVector<f64> {
  let b_lo: DVector<f64> = DVector::from_element(problem.dim(), problem.domain().0);
  let b_up: DVector<f64> = DVector::from_element(problem.dim(), problem.domain().1);

  uniform_distribution(
    &DVector::from_iterator(problem.dim(), (&b_up - &b_lo).iter().map(|b| -b.abs())),
    &DVector::from_iterator(problem.dim(), (&b_up - &b_lo).iter().map(|b| b.abs())),
  )
}

pub fn create_directory(path: PathBuf) {
  // Handle output directory creation / deletion
  if path.exists() {
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

pub fn grid_search<'a, U: ParticleTrait, T: PSOTrait<'a, U>>(
  particle_count: usize,
  iterations: usize,
  problem_set: &'a Vec<OptimizationProblem>,
  attempts: usize,
  search_params: Vec<(String, (f64, f64))>,
  base_params: HashMap<String, f64>,
  out_directory: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
  assert!(search_params.len() == 2);
  create_directory(out_directory.clone());
  let steps: usize = 20;
  for problem in problem_set {
    let bar = ProgressBar::new(((steps + 1) * (steps + 1) * attempts) as u64);
    for p in 0..=steps {
      for g in 0..=steps {
        let p = p as f64 / steps as f64 * (search_params[0].1 .1 - search_params[0].1 .0) + search_params[0].1 .0;
        let g = g as f64 / steps as f64 * (search_params[1].1 .1 - search_params[1].1 .0) + search_params[1].1 .0;
        for attempt in 0..attempts {
          let mut params = base_params.clone();
          params.insert(search_params[0].0.clone(), p);
          params.insert(search_params[1].0.clone(), g);
          let mut pso: T = T::new(
            "PSO",
            &problem,
            particle_count,
            params,
            out_directory.join(format!(
              "{}/{}={:.2},{}={:.2}/{}",
              problem.name(),
              search_params[0].0.clone(),
              p,
              search_params[1].0.clone(),
              g,
              attempt
            )),
          );
          pso.run(iterations);
          pso.save_summary()?;
          pso.save_config()?;
          bar.inc(1);
        }
      }
    }
    bar.finish();
  }
  Ok(())
}
