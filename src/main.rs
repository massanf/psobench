extern crate nalgebra as na;
extern crate rand;
mod problem;
use std::path::PathBuf;
mod functions;
mod particle_trait;
use crate::pso_trait::ParamValue;
use std::collections::HashMap;
mod grid_search;
mod gsa;
mod pso;
mod pso_trait;
use crate::pso_trait::PSOTrait;
use gsa::gsa::GSA;
use gsa::particle::GSAParticle;
use pso::particle::Particle;
use pso::pso::PSO;
mod utils;

#[allow(dead_code)]
fn run_pso() -> Result<(), Box<dyn std::error::Error>> {
  // Experiment Settings
  let iterations = 1000;
  let params: HashMap<String, ParamValue> = [
    ("w".to_owned(), ParamValue::Float(0.8)),
    ("phi_p".to_owned(), ParamValue::Float(1.0)),
    ("phi_g".to_owned(), ParamValue::Float(1.0)),
    ("particle_count".to_owned(), ParamValue::Int(30)),
  ]
  .iter()
  .cloned()
  .collect();

  let mut pso: PSO<Particle> = PSO::new("PSO", functions::f1(10), params.clone(), PathBuf::from("data/test"));
  pso.run(iterations);
  pso.save_summary()?;
  pso.save_data()?;
  pso.save_config(&params)?;
  Ok(())
}

#[allow(dead_code)]
fn run_grid_search() -> Result<(), Box<dyn std::error::Error>> {
  let iterations = 1000;
  let params: HashMap<String, ParamValue> = [
    ("particle_count".to_owned(), ParamValue::Int(30)),
    ("g0".to_owned(), ParamValue::Float(1.)),
    ("alpha".to_owned(), ParamValue::Float(20.)),
    ("epsilon".to_owned(), ParamValue::Float(0.00000001)),
  ]
  .iter()
  .cloned()
  .collect();

  let mut gsa: GSA<GSAParticle> = GSA::new("GSA", functions::f1(5), params.clone(), PathBuf::from("data/bad"));
  gsa.run(iterations);
  gsa.save_summary()?;
  gsa.save_config(&params)?;
  gsa.save_data()?;

  let out_directory = PathBuf::from("data/base_gsa_all");

  let g: Vec<ParamValue> = vec![
    ParamValue::Float(0.1),
    ParamValue::Float(0.2),
    ParamValue::Float(0.5),
    ParamValue::Float(1.0),
    ParamValue::Float(2.0),
    ParamValue::Float(5.0),
    ParamValue::Float(10.0),
    ParamValue::Float(20.0),
    ParamValue::Float(50.0),
    ParamValue::Float(100.0),
  ];
  let alpha: Vec<ParamValue> = vec![
    ParamValue::Float(0.1),
    ParamValue::Float(0.2),
    ParamValue::Float(0.5),
    ParamValue::Float(1.0),
    ParamValue::Float(2.0),
    ParamValue::Float(5.0),
    ParamValue::Float(10.0),
    ParamValue::Float(20.0),
    ParamValue::Float(50.0),
    ParamValue::Float(100.0),
  ];

  let base_params: HashMap<String, ParamValue> = [
    ("particle_count".to_owned(), ParamValue::Int(30)),
    ("epsilon".to_owned(), ParamValue::Float(0.001)),
  ]
  .iter()
  .cloned()
  .collect();

  for func_num in 1..=30 {
    if func_num == 2 {
      continue;
    }
    grid_search::grid_search::<GSAParticle, GSA<GSAParticle>>(
      iterations,
      functions::cec17(func_num, 30),
      5,
      ("g0".to_owned(), g.clone()),
      ("alpha".to_owned(), alpha.clone()),
      base_params.clone(),
      out_directory.clone(),
    )?;
  }
  Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  run_grid_search()?;
  Ok(())
}
