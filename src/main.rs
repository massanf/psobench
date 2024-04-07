extern crate nalgebra as na;
extern crate rand;
mod problem;
use crate::pso_trait::DataExporter;
use std::path::PathBuf;
mod functions;
mod particle_trait;
use crate::pso_trait::ParamValue;
use std::collections::HashMap;
mod fdo;
mod grid_search;
mod gsa;
mod pso;
mod pso_trait;
mod tiled_gsa;
use crate::pso_trait::ParticleOptimizer;
use crate::tiled_gsa::tiled_gsa::TiledGSA;
use fdo::fdo::FDO;
use fdo::particle::FDOParticle;
use gsa::gsa::GSA;
use gsa::particle::GSAParticle;
use pso::particle::PSOParticle;
use pso::pso::PSO;
use tiled_gsa::particle::TiledGSAParticle;
mod utils;

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
    "PSO",
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
    ("g0".to_owned(), ParamValue::Float(100.0)),
    ("alpha".to_owned(), ParamValue::Float(20.0)),
    ("particle_count".to_owned(), ParamValue::Int(50)),
  ]
  .iter()
  .cloned()
  .collect();

  let mut gsa: GSA<GSAParticle> = GSA::new(
    "GSA",
    functions::cec17(1, 10),
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
    ("particle_count".to_owned(), ParamValue::Int(50)),
  ]
  .iter()
  .cloned()
  .collect();

  let mut gsa: TiledGSA<TiledGSAParticle> = TiledGSA::new(
    "TiledGSA",
    functions::cec17(1, 10),
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
    ("particle_count".to_owned(), ParamValue::Int(50)),
    ("wf".to_owned(), ParamValue::Int(1)),
  ]
  .iter()
  .cloned()
  .collect();

  let mut fdo: FDO<FDOParticle> = FDO::new(
    "FDO",
    // functions::f1(10),
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
      5,
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

    grid_search::grid_search::<TiledGSAParticle, TiledGSA<TiledGSAParticle>>(
      "TiledGSA".to_owned(),
      iterations,
      functions::cec17(func_num, dim),
      5,
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
      5,
      ("phi_p".to_owned(), phi_p.clone()),
      ("phi_g".to_owned(), phi_g.clone()),
      base_params.clone(),
      out_directory.clone(),
    )?;
  }
  Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  run_grid_search_gsa(30)?;
  // run_grid_search_tiled_gsa(30)?;
  run_grid_search_pso(30)?;
  // run_cfo()?;
  // run_gsa()?;
  // run_fdo()?;
  // run_tiled_gsa()?;
  // run_pso()?;
  Ok(())
}
