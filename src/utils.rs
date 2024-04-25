use crate::grid_search;
use crate::optimizers::traits::{DataExporter, Optimizer, ParamValue};
use crate::particles::traits::{Behavior, Position, Velocity};
use crate::problems;
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::DVector;
use problems::Problem;
extern crate chrono;
use rand::distributions::{Distribution, Uniform};
use rayon::prelude::*;
use std::{collections::HashMap, fs, io, path::PathBuf};

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

        if user_input.trim().to_lowercase().as_str() == "y" {
          let _ = fs::remove_dir_all(path.clone());
          let _ = fs::create_dir_all(path);
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

#[allow(clippy::too_many_arguments)]
pub fn run_attempts<U: Position + Velocity + Clone, T: Optimizer<U> + DataExporter<U>>(
  params: HashMap<String, ParamValue>,
  name: String,
  problem: Problem,
  out_directory: PathBuf,
  iterations: usize,
  attempts: usize,
  save_data: bool,
  bar: &indicatif::ProgressBar,
  behavior: Behavior,
) -> Result<(), Box<dyn std::error::Error>> {
  (0..attempts).into_par_iter().for_each(|attempt| {
    let save = save_data && attempt < 1;
    let mut pso: T = T::new(
      name.clone(),
      problem.clone(),
      params.clone(),
      out_directory.join(format!("{}", attempt)),
      behavior,
      save,
    );
    pso.run(iterations);
    let _ = pso.save_summary();
    let _ = pso.save_config(&params);
    if save {
      let _ = pso.save_data();
    }
    bar.inc(1);
  });
  Ok(())
}

#[allow(dead_code)]
pub fn check_cec17<T: Velocity + Clone, U: Optimizer<T>>(
  test_name: &str,
  optimizer_name: &str,
  iterations: usize,
  dim: usize,
  attempts: usize,
  behavior: Behavior,
  params_in_vec: Vec<(&str, ParamValue)>,
  // out_directory: String,
) -> Result<(), Box<dyn std::error::Error>> {
  let params = param_hashmap_generator(params_in_vec);
  // Progress Bar.
  let bar = ProgressBar::new((29 * attempts) as u64);
  bar.set_style(
    ProgressStyle::default_bar()
      .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {msg}")
      .unwrap()
      .progress_chars("#>-"),
  );
  bar.set_message(format!("{}...   ", optimizer_name));

  let mut func_nums = Vec::new();
  for func_num in 1..=30 {
    if func_num == 2 {
      continue;
    }
    func_nums.push(func_num);
  }

  let out_directory = generate_out_directory(test_name, dim, optimizer_name);

  func_nums.into_par_iter().for_each(|func_num: usize| {
    let problem = problems::cec17(func_num, dim);
    let _ = run_attempts::<T, U>(
      params.clone(),
      optimizer_name.to_owned().clone(),
      problem.clone(),
      out_directory.join(problem.clone().name()),
      iterations,
      attempts,
      false,
      &bar,
      behavior,
    );
  });

  Ok(())
}

#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub fn run_grid_searches<T: Velocity + Clone, U: Optimizer<T>>(
  optimizer_name: &str,
  attempts: usize,
  iterations: usize,
  dim: usize,
  param1: (String, Vec<ParamValue>),
  param2: (String, Vec<ParamValue>),
  base_params_in_vec: Vec<(&str, ParamValue)>,
  behavior: Behavior,
) -> Result<(), Box<dyn std::error::Error>> {
  let base_params = param_hashmap_generator(base_params_in_vec);
  let out_directory = generate_out_directory("grid_search", dim, optimizer_name);

  for func_num in 1..=30 {
    if func_num == 2 {
      continue;
    }

    grid_search::grid_search::<T, U>(
      optimizer_name.to_owned().clone(),
      iterations,
      problems::cec17(func_num, dim),
      attempts,
      param1.clone(),
      param2.clone(),
      base_params.clone(),
      out_directory.clone(),
      behavior,
    )?;
  }
  Ok(())
}

#[allow(dead_code)]
pub fn param_hashmap_generator(params: Vec<(&str, ParamValue)>) -> HashMap<String, ParamValue> {
  let mut vec = Vec::new();
  for param in params {
    vec.push((param.0.to_owned(), param.1));
  }
  vec.iter().cloned().collect()
}

#[allow(dead_code)]
pub fn generate_out_directory(test_name: &str, dim: usize, type_name: &str) -> PathBuf {
  PathBuf::from(format!("data/{}/{}/{}", test_name, dim, type_name))
}

pub fn min_max_normalize(input: Vec<f64>) -> Vec<f64> {
  if input.is_empty() {
    return Vec::new(); // Return an empty vector if input is empty
  }

  let min = input.iter().fold(f64::INFINITY, |a, &b| a.min(b));
  let max = input.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

  if (max - min).abs() < f64::EPSILON {
    return vec![0.0; input.len()]; // Return zero vector if all elements are the same
  }

  input.into_iter().map(|x| (x - max) / (min - max)).collect()
}

pub fn z_score_normalize(input: Vec<f64>) -> Vec<f64> {
  let mean = input.iter().sum::<f64>() / input.len() as f64;
  let std = (input.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / input.len() as f64).sqrt();

  input.iter().map(|&x| (x - mean) / std).collect()
}

pub fn original_gsa_normalize(input: Vec<f64>) -> Vec<f64> {
  let input = min_max_normalize(input);
  let sum: f64 = input.iter().sum();
  input.iter().map(|&x| x / sum).collect()
}

pub fn sigmoid_normalize(input: Vec<f64>) -> Vec<f64> {
  z_score_normalize(input).into_iter().map(|x| 1.0 / (1.0 + (-2. * x).exp())).collect()
}

pub fn softmax_normalize(input: Vec<f64>) -> Vec<f64> {
  let max = input.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
  let exps: Vec<f64> = input.iter().map(|&x| ((x - max).exp()).min(f64::MAX)).collect();
  let sum_exps: f64 = exps.iter().sum();

  exps.iter().map(|&exp| exp / sum_exps).collect()
}

pub fn rank_normalize(input: Vec<f64>) -> Vec<f64> {
  let mut sorted_input: Vec<_> = input.iter().enumerate().collect();
  sorted_input.sort_by(|i, j| i.1.partial_cmp(j.1).unwrap());
  let mut mp: HashMap<usize, usize> = HashMap::new();
  for (idx, (i, _x)) in sorted_input.iter().enumerate() {
    mp.insert(*i, idx);
  }
  input.iter().enumerate().map(|(i, _)| mp[&i] as f64 / input.len() as f64).collect()
}
