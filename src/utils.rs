use crate::grid_search;
use crate::optimizers::traits::{DataExporter, Optimizer, ParamValue};
use crate::particles::traits::{Behavior, Edge, Position, Velocity};
use crate::problems;
use crate::Normalizer;
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
) -> Result<(), Box<dyn std::error::Error>> {
  (0..attempts).into_par_iter().for_each(|attempt| {
    let save = save_data && attempt < 1;
    let mut pso: T = T::new(
      name.clone(),
      problem.clone(),
      params.clone(),
      out_directory.join(format!("{}", attempt)),
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

#[allow(clippy::too_many_arguments)]
pub fn check_cec17<T: Velocity + Clone, U: Optimizer<T>>(
  test_name: &str,
  optimizer_name: &str,
  iterations: usize,
  dim: usize,
  attempts: usize,
  params_in_vec: Vec<(&str, ParamValue)>,
  save: bool,
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
      save,
      &bar,
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
    )?;
  }
  Ok(())
}

pub fn param_hashmap_generator(params: Vec<(&str, ParamValue)>) -> HashMap<String, ParamValue> {
  let mut vec = Vec::new();
  for param in params {
    vec.push((param.0.to_owned(), param.1));
  }
  vec.iter().cloned().collect()
}

pub fn generate_out_directory(test_name: &str, dim: usize, type_name: &str) -> PathBuf {
  PathBuf::from(format!("data/{}/{}/{}", test_name, dim, type_name))
}

pub fn quantile(input: &[f64], q: f64) -> f64 {
  // This is approximate; should be good enough.
  let mut sorted = input.to_owned();
  sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
  sorted[(sorted.len() as f64 * q) as usize]
}

// Normalizers
pub fn min_max_normalize(input: Vec<f64>) -> Vec<f64> {
  let min = input.iter().fold(f64::INFINITY, |a, &b| a.min(b));
  let max = input.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
  if (max - min).abs() < f64::EPSILON {
    return vec![0.5; input.len()]; // Return zero vector if all elements are the same
  }
  input.into_iter().map(|x| (x - min) / (max - min)).collect()
}

pub fn robust_normalize(input: Vec<f64>) -> Vec<f64> {
  let med = quantile(&input, 0.5);
  let niqr = (quantile(&input, 0.75) - quantile(&input, 0.25)) / 1.3489;
  if niqr < f64::EPSILON {
    return vec![0.0; input.len()];
  }
  input.iter().map(|x| (x - med) / niqr).collect()
}

pub fn z_score_normalize(input: Vec<f64>) -> Vec<f64> {
  let mean = input.iter().sum::<f64>() / input.len() as f64;
  let std = (input.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / input.len() as f64).sqrt();
  if std.abs() < f64::EPSILON {
    return vec![0.0; input.len()];
  }
  input.iter().map(|&x| (x - mean) / std).collect()
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

fn sigmoid_normalize(input: Vec<f64>, scale: f64) -> Vec<f64> {
  z_score_normalize(input).into_iter().map(|x| 1.0 / (1.0 + (-1. * scale * x).exp())).collect()
}

// mass
pub fn original_gsa_mass(input: Vec<f64>) -> Vec<f64> {
  let input: Vec<f64> = min_max_normalize(input).iter().map(|x| 1. - x).collect();
  let sum: f64 = input.iter().sum();
  if sum.abs() < f64::EPSILON {
    return vec![0.0; input.len()];
  }
  input.iter().map(|&x| x / sum).collect()
}

pub fn z_mass(input: Vec<f64>) -> Vec<f64> {
  z_score_normalize(input).iter().map(|x| -0.5 * x).map(|x| if x < 0. { 0. } else { x }).collect()
}

pub fn robust_mass(input: Vec<f64>) -> Vec<f64> {
  robust_normalize(input).iter().map(|x| -0.5 * x).map(|x| if x < 0. { 0. } else { x }).collect()
}

pub fn sigmoid2_mass(input: Vec<f64>) -> Vec<f64> {
  sigmoid_normalize(input, -2.)
}

pub fn sigmoid4_mass(input: Vec<f64>) -> Vec<f64> {
  sigmoid_normalize(input, -4.)
}

pub fn rank_mass(input: Vec<f64>) -> Vec<f64> {
  rank_normalize(input).iter().map(|x| 1. - x).collect()
}

// Parameter getters
#[allow(dead_code)]
pub fn behavior_from_tiled(tiled: bool) -> ParamValue {
  ParamValue::Behavior(Behavior {
    edge: match tiled {
      true => Edge::Cycle,
      false => Edge::Pass,
    },
    vmax: false,
  })
}

#[allow(dead_code)]
pub fn g0_from_normalizer(normalizer: Normalizer) -> ParamValue {
  ParamValue::Float(match normalizer {
    Normalizer::MinMax => 1000.,
    Normalizer::ZScore => 100.,
    _ => 50.,
  })
}

#[allow(dead_code)]
pub fn alpha_from_normalizer(_normalizer: Normalizer) -> ParamValue {
  ParamValue::Float(5.)
}

#[allow(dead_code)]
pub fn name_from_normalizer_and_tiled(normalizer: Normalizer, tiled: bool) -> String {
  match tiled {
    true => format!("gsa_{:?}_tiled", normalizer),
    false => format!("gsa_{:?}", normalizer),
  }
}
