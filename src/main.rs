extern crate nalgebra as na;
use indicatif::ProgressBar;
extern crate rand;
use rayon::prelude::*;
mod problem;
use crate::pso_trait::DataExporter;
use std::path::PathBuf;
mod functions;
mod particle_trait;
use crate::pso_trait::ParamValue;
use std::collections::HashMap;
mod fdo;
mod grid_search;
use indicatif::ProgressStyle;
mod gsa;
mod pso;
mod pso_trait;
use crate::gsa::tiled_gsa::TiledGSA;
use crate::pso_trait::ParticleOptimizer;
use fdo::fdo::FDO;
use fdo::particle::FDOParticle;
use gsa::gsa::GSA;
use gsa::particle::GSAParticle;
use gsa::tiled_gsa_particle::TiledGSAParticle;
use pso::particle::PSOParticle;
use pso::pso::PSO;
mod utils;
use crate::particle_trait::Velocity;

#[allow(dead_code)]
fn check_cec17<T: Velocity, U: ParticleOptimizer<T>>(
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
    let _ = utils::run_attempts::<T, U>(
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
fn run_pso() -> Result<(), Box<dyn std::error::Error>> {
  // Experiment Settings
  let iterations = 1000;
  let params: HashMap<String, ParamValue> = [
    ("w".to_owned(), ParamValue::Float(0.8)),
    ("phi_p".to_owned(), ParamValue::Float(1.0)),
    ("phi_g".to_owned(), ParamValue::Float(1.0)),
    ("particle_count".to_owned(), ParamValue::Int(50)),
  ]
  .iter()
  .cloned()
  .collect();

  let mut pso: PSO<PSOParticle> = PSO::new(
    "PSO".to_owned(),
    functions::cec17(1, 10),
    params.clone(),
    PathBuf::from("data/test/pso"),
  );
  pso.run(iterations);
  pso.save_summary()?;
  pso.save_data()?;
  pso.save_config(&params)?;
  Ok(())
}

#[allow(dead_code)]
fn run_gsa() -> Result<(), Box<dyn std::error::Error>> {
  // Experiment Settings
  let iterations = 1000;
  let params: HashMap<String, ParamValue> = [
    ("g0".to_owned(), ParamValue::Float(5.0)),
    ("alpha".to_owned(), ParamValue::Float(20.0)),
    ("particle_count".to_owned(), ParamValue::Int(30)),
  ]
  .iter()
  .cloned()
  .collect();

  let mut gsa: GSA<GSAParticle> = GSA::new(
    "GSA".to_owned(),
    // functions::cec17(1, 10),
    functions::f1(30),
    params.clone(),
    PathBuf::from("data/test/gsa"),
  );
  gsa.run(iterations);
  gsa.save_summary()?;
  gsa.save_data()?;
  gsa.save_config(&params)?;
  Ok(())
}

#[allow(dead_code)]
fn run_tiled_gsa() -> Result<(), Box<dyn std::error::Error>> {
  // Experiment Settings
  let iterations = 1000;
  let params: HashMap<String, ParamValue> = [
    ("g0".to_owned(), ParamValue::Float(100.0)),
    ("alpha".to_owned(), ParamValue::Float(20.0)),
    ("particle_count".to_owned(), ParamValue::Int(100)),
  ]
  .iter()
  .cloned()
  .collect();

  let mut gsa: TiledGSA<TiledGSAParticle> = TiledGSA::new(
    "TiledGSA".to_owned(),
    functions::cec17(1, 30),
    params.clone(),
    PathBuf::from("data/test/tiled_gsa"),
  );
  gsa.run(iterations);
  gsa.save_summary()?;
  gsa.save_data()?;
  gsa.save_config(&params)?;
  Ok(())
}

#[allow(dead_code)]
fn run_fdo() -> Result<(), Box<dyn std::error::Error>> {
  // Experiment Settings
  let iterations = 1000;
  let params: HashMap<String, ParamValue> = [
    ("particle_count".to_owned(), ParamValue::Int(30)),
    ("wf".to_owned(), ParamValue::Int(1)),
  ]
  .iter()
  .cloned()
  .collect();

  let mut fdo: FDO<FDOParticle> = FDO::new(
    "FDO".to_owned(),
    functions::cec17(1, 100),
    params.clone(),
    PathBuf::from("data/test/fdo"),
  );
  fdo.run(iterations);
  fdo.save_summary()?;
  fdo.save_data()?;
  fdo.save_config(&params)?;
  Ok(())
}

#[allow(dead_code)]
fn run_grid_search_gsa(dim: usize) -> Result<(), Box<dyn std::error::Error>> {
  let iterations = 1000;
  let out_directory = PathBuf::from(format!("data/grid_search/gsa_{}", dim));

  let g0: Vec<ParamValue> = vec![
    ParamValue::Float(10.0),
    ParamValue::Float(50.0),
    ParamValue::Float(100.0),
    ParamValue::Float(500.0),
    ParamValue::Float(1000.0),
    ParamValue::Float(5000.0),
  ];
  let alpha: Vec<ParamValue> = vec![
    ParamValue::Float(1.0),
    ParamValue::Float(2.0),
    ParamValue::Float(5.0),
    ParamValue::Float(10.0),
    ParamValue::Float(20.0),
    ParamValue::Float(50.0),
    ParamValue::Float(100.0),
  ];

  let base_params: HashMap<String, ParamValue> =
    [("particle_count".to_owned(), ParamValue::Int(30))].iter().cloned().collect();

  for func_num in 1..=30 {
    if func_num == 2 {
      continue;
    }

    grid_search::grid_search::<GSAParticle, GSA<GSAParticle>>(
      "GSA".to_owned(),
      iterations,
      functions::cec17(func_num, dim),
      10,
      ("g0".to_owned(), g0.clone()),
      ("alpha".to_owned(), alpha.clone()),
      base_params.clone(),
      out_directory.clone(),
    )?;
  }
  Ok(())
}

#[allow(dead_code)]
fn run_grid_search_tiled_gsa(dim: usize) -> Result<(), Box<dyn std::error::Error>> {
  let iterations = 1000;
  let out_directory = PathBuf::from(format!("data/grid_search/tiled_gsa_{}", dim));

  let g0: Vec<ParamValue> = vec![
    ParamValue::Float(100.0),
    ParamValue::Float(500.0),
    ParamValue::Float(1000.0),
    ParamValue::Float(5000.0),
    ParamValue::Float(10000.0),
    ParamValue::Float(50000.0),
  ];
  let alpha: Vec<ParamValue> = vec![
    ParamValue::Float(0.1),
    ParamValue::Float(1.0),
    ParamValue::Float(2.0),
    ParamValue::Float(5.0),
    ParamValue::Float(10.0),
    ParamValue::Float(20.0),
    ParamValue::Float(50.0),
    ParamValue::Float(100.0),
  ];

  let base_params: HashMap<String, ParamValue> =
    [("particle_count".to_owned(), ParamValue::Int(30))].iter().cloned().collect();

  for func_num in 1..=30 {
    if func_num == 2 {
      continue;
    }

    grid_search::grid_search::<TiledGSAParticle, TiledGSA<TiledGSAParticle>>(
      "TiledGSA".to_owned(),
      iterations,
      functions::cec17(func_num, dim),
      10,
      ("g0".to_owned(), g0.clone()),
      ("alpha".to_owned(), alpha.clone()),
      base_params.clone(),
      out_directory.clone(),
    )?;
  }
  Ok(())
}

#[allow(dead_code)]
fn run_grid_search_pso(dim: usize) -> Result<(), Box<dyn std::error::Error>> {
  let iterations = 1000;
  let out_directory = PathBuf::from(format!("data/grid_search/pso_{}", dim));

  let phi_p: Vec<ParamValue> = vec![
    ParamValue::Float(-4.0),
    ParamValue::Float(-3.0),
    ParamValue::Float(-2.0),
    ParamValue::Float(-1.0),
    ParamValue::Float(0.0),
    ParamValue::Float(1.0),
    ParamValue::Float(2.0),
    ParamValue::Float(3.0),
    ParamValue::Float(4.0),
  ];

  let phi_g: Vec<ParamValue> = vec![
    ParamValue::Float(-4.0),
    ParamValue::Float(-3.0),
    ParamValue::Float(-2.0),
    ParamValue::Float(-1.0),
    ParamValue::Float(0.0),
    ParamValue::Float(1.0),
    ParamValue::Float(2.0),
    ParamValue::Float(3.0),
    ParamValue::Float(4.0),
  ];

  let base_params: HashMap<String, ParamValue> = [
    ("w".to_owned(), ParamValue::Float(0.8)),
    ("particle_count".to_owned(), ParamValue::Int(30)),
  ]
  .iter()
  .cloned()
  .collect();

  for func_num in 1..=30 {
    if func_num == 2 {
      continue;
    }

    grid_search::grid_search::<PSOParticle, PSO<PSOParticle>>(
      "PSO".to_owned(),
      iterations,
      functions::cec17(func_num, dim),
      10,
      ("phi_p".to_owned(), phi_p.clone()),
      ("phi_g".to_owned(), phi_g.clone()),
      base_params.clone(),
      out_directory.clone(),
    )?;
  }
  Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let dim = 30;
  let iterations = 1000;
  let attempts = 30;

  // let pso_params: HashMap<String, ParamValue> = [
  //   ("w".to_owned(), ParamValue::Float(0.8)),
  //   ("phi_p".to_owned(), ParamValue::Float(1.0)),
  //   ("phi_g".to_owned(), ParamValue::Float(1.0)),
  //   ("particle_count".to_owned(), ParamValue::Int(30)),
  // ]
  // .iter()
  // .cloned()
  // .collect();
  // check_cec17::<PSOParticle, PSO<PSOParticle>>(
  //   "PSO".to_owned(),
  //   iterations,
  //   dim,
  //   pso_params,
  //   attempts,
  //   PathBuf::from(format!("data/test/pso_{}", dim)),
  // )?;

  // let gsa_params: HashMap<String, ParamValue> = [
  //   ("g0".to_owned(), ParamValue::Float(5000.0)),
  //   ("alpha".to_owned(), ParamValue::Float(5.0)),
  //   ("particle_count".to_owned(), ParamValue::Int(30)),
  // ]
  // .iter()
  // .cloned()
  // .collect();
  // check_cec17::<GSAParticle, GSA<GSAParticle>>(
  //   "GSA".to_owned(),
  //   iterations,
  //   dim,
  //   gsa_params,
  //   attempts,
  //   PathBuf::from(format!("data/test/gsa_{}", dim)),
  // )?;

  let tiled_gsa_params: HashMap<String, ParamValue> = [
    ("g0".to_owned(), ParamValue::Float(1000.0)),
    ("alpha".to_owned(), ParamValue::Float(5.0)),
    ("particle_count".to_owned(), ParamValue::Int(30)),
  ]
  .iter()
  .cloned()
  .collect();
  check_cec17::<TiledGSAParticle, TiledGSA<TiledGSAParticle>>(
    "TiledGSA".to_owned(),
    iterations,
    dim,
    tiled_gsa_params,
    attempts,
    PathBuf::from(format!("data/test/tiled_gsa_{}", dim)),
  )?;

  // run_grid_search_gsa(30)?;
  // run_grid_search_tiled_gsa(30)?;
  // run_grid_search_pso(30)?;
  // run_cfo()?;
  // run_gsa()?;
  // run_fdo()?;
  // run_tiled_gsa()?;
  // run_pso()?;
  Ok(())
}
