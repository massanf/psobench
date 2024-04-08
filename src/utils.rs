use crate::functions;
use crate::grid_search;
use crate::particle_trait::{Position, Velocity};
use crate::problem;
use crate::DataExporter;
use crate::ParamValue;
use crate::ParticleOptimizer;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
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

#[allow(dead_code)]
pub fn check_cec17<T: Velocity, U: ParticleOptimizer<T>>(
  name: String,
  iterations: usize,
  dim: usize,
  params: HashMap<String, ParamValue>,
  attempts: usize,
  out_directory: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
  // Progress Bar.
  let bar = ProgressBar::new((29 * attempts) as u64);
  bar.set_style(
    ProgressStyle::default_bar()
      .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {msg}")
      .unwrap()
      .progress_chars("#>-"),
  );
  bar.set_message(format!("{}...   ", name.clone()));

  let mut func_nums = Vec::new();
  for func_num in 1..=30 {
    if func_num == 2 {
      continue;
    }
    func_nums.push(func_num);
  }

  func_nums.into_par_iter().for_each(|func_num: usize| {
    let problem = functions::cec17(func_num, dim);
    let _ = run_attempts::<T, U>(
      params.clone(),
      name.clone(),
      problem.clone(),
      out_directory.join(format!("{}", problem.clone().name())),
      iterations,
      attempts,
      true,
      &bar,
    );
  });

  Ok(())
}

#[allow(dead_code)]
pub fn run_grid_searches<T: Velocity, U: ParticleOptimizer<T>>(
  name: String,
  attempts: usize,
  param1: (String, Vec<ParamValue>),
  param2: (String, Vec<ParamValue>),
  base_params: HashMap<String, ParamValue>,
  dim: usize,
  out_directory: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
  let iterations = 1000;

  for func_num in 1..=30 {
    if func_num == 2 {
      continue;
    }

    grid_search::grid_search::<T, U>(
      name.clone(),
      iterations,
      functions::cec17(func_num, dim),
      attempts,
      param1.clone(),
      param2.clone(),
      base_params.clone(),
      out_directory.clone(),
    )?;
  }
  Ok(())
}
