use crate::optimization_problem;
use crate::particle_trait::ParticleTrait;
use crate::pso_trait::PSOTrait;
use crate::pso_trait::ParamValue;
use indicatif::ProgressBar;
use nalgebra::DVector;
use optimization_problem::Problem;
use rand::distributions::{Distribution, Uniform};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

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

fn run_attempts<U: ParticleTrait, T: PSOTrait<U>>(
  params: HashMap<String, ParamValue>,
  problem: Problem,
  out_directory: PathBuf,
  iterations: usize,
  attempts: usize,
  bar: &indicatif::ProgressBar,
) -> Result<(), Box<dyn std::error::Error>> {
  for attempt in 0..attempts {
    let mut pso: T = T::new(
      "PSO",
      problem.clone(),
      params.clone(),
      out_directory.join(format!("{}", attempt)),
    );
    pso.run(iterations);
    pso.save_summary()?;
    pso.save_config()?;
    bar.inc(1);
  }
  Ok(())
}

#[allow(dead_code)]
pub fn grid_search<U: ParticleTrait, T: PSOTrait<U>>(
  iterations: usize,
  problem: Problem,
  attempts: usize,
  param1: (String, Vec<ParamValue>),
  param2: (String, Vec<ParamValue>),
  base_params: HashMap<String, ParamValue>,
  out_folder: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
  create_directory(out_folder.clone(), false);
  let out_directory = out_folder.join(problem.name());
  create_directory(out_directory.clone(), true);

  let bar = ProgressBar::new(((param1.1.len()) * (param2.1.len()) * attempts) as u64);
  for x1 in &param1.1 {
    for x2 in &param2.1 {
      let mut params = base_params.clone();
      params.insert(param1.0.clone(), x1.clone());
      params.insert(param2.0.clone(), x2.clone());

      run_attempts::<U, T>(
        params,
        problem.clone(),
        out_directory.join(format!("{}={},{}={}", param1.0, x1, param2.0, x2)),
        iterations,
        attempts,
        &bar,
      )?;
    }
  }
  Ok(())
}

#[allow(dead_code)]
pub fn grid_search_dim<U: ParticleTrait, T: PSOTrait<U>>(
  iterations: usize,
  problem_type: Arc<dyn Fn(usize) -> Problem>,
  attempts: usize,
  dims: Vec<usize>,
  param: (String, Vec<ParamValue>),
  base_params: HashMap<String, ParamValue>,
  out_folder: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
  create_directory(out_folder.clone(), false);
  let out_directory = out_folder.join(problem_type(2).name());
  create_directory(out_directory.clone(), true);

  let bar = ProgressBar::new((dims.len() * param.1.len() * attempts) as u64);
  for x in param.1 {
    for dim in &dims {
      let problem = problem_type(dim.clone());
      let mut params = base_params.clone();
      params.insert(param.0.clone(), x.clone());

      run_attempts::<U, T>(
        params,
        problem.clone(),
        out_directory.join(format!("{}={},{}={}", param.0.clone(), x, "dim", dim)),
        iterations,
        attempts,
        &bar,
      )?;
    }
  }
  Ok(())
}
