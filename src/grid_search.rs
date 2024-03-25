use crate::optimization_problem::Problem;
use crate::particle_trait::ParticleTrait;
use crate::pso_trait::{PSOTrait, ParamValue};
use crate::utils;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde_json::json;
use std::{collections::HashMap, fs, path::PathBuf, sync::Arc};

fn run_attempts<U: ParticleTrait, T: PSOTrait<U>>(
  params: HashMap<String, ParamValue>,
  problem: Problem,
  out_directory: PathBuf,
  iterations: usize,
  attempts: usize,
  bar: &indicatif::ProgressBar,
) -> Result<(), Box<dyn std::error::Error>> {
  (0..attempts).into_par_iter().for_each(|attempt| {
    let mut pso: T = T::new(
      "PSO",
      problem.clone(),
      params.clone(),
      out_directory.join(format!("{}", attempt)),
    );
    pso.run(iterations);
    let _ = pso.save_summary();
    let _ = pso.save_config();
    bar.inc(1);
  });
  Ok(())
}

fn save_grid_search_config(
  problem: Problem,
  param1: (String, Vec<ParamValue>),
  param2: (String, Vec<ParamValue>),
  out_directory: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
  let serialized = serde_json::to_string(&json!({
    "problem": {
      "name": problem.name(),
      "dim": problem.dim(),
    },
    "grid_search": {
      param1.0: param1.1,
      param2.0: param2.1,
  }  }))?;
  fs::write(out_directory.join("grid_search_config.json"), serialized)?;
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
  utils::create_directory(out_folder.clone(), true);
  let out_directory = out_folder.join(problem.name());
  utils::create_directory(out_directory.clone(), false);

  let bar = ProgressBar::new(((param1.1.len()) * (param2.1.len()) * attempts) as u64);
  bar.set_style(
    ProgressStyle::default_bar()
      .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {msg}")
      .unwrap()
      .progress_chars("#>-"),
  );
  bar.set_message(format!("{}...", problem.name()));
  for x1 in &param1.1 {
    for x2 in &param2.1 {
      let mut params: HashMap<String, ParamValue> = base_params.clone();
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

  bar.finish_with_message(format!("{} done!", problem.name()));
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
  utils::create_directory(out_folder.clone(), true);
  let out_directory = out_folder.join(problem_type(2).name());
  utils::create_directory(out_directory.clone(), false);

  let mut dim_param: Vec<ParamValue> = Vec::new();
  for d in &dims {
    dim_param.push(ParamValue::Int(*d as isize));
  }
  let bar = ProgressBar::new((dims.len() * param.1.len() * attempts) as u64);
  bar.set_style(
    ProgressStyle::default_bar()
      .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} {msg}")
      .unwrap()
      .progress_chars("#>-"),
  );
  bar.set_message(format!("{}...", problem_type(2).name()));
  for x in &param.1 {
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
  bar.finish_with_message(format!("{} done!", problem_type(2).name()));

  save_grid_search_config(problem_type(2), ("dim".to_owned(), dim_param), param, out_directory)?;
  Ok(())
}
