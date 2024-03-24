use crate::optimization_problem::Problem;
use crate::particle_trait::ParticleTrait;
use crate::pso_trait::PSOTrait;
use crate::pso_trait::ParamValue;
use crate::utils;
use indicatif::ProgressBar;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

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
  utils::create_directory(out_folder.clone(), false);
  let out_directory = out_folder.join(problem.name());
  utils::create_directory(out_directory.clone(), true);

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
  utils::create_directory(out_folder.clone(), false);
  let out_directory = out_folder.join(problem_type(2).name());
  utils::create_directory(out_directory.clone(), true);

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
