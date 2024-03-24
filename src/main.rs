extern crate nalgebra as na;
extern crate rand;
mod optimization_problem;
use std::path::PathBuf;
mod functions;
mod particle_trait;
use crate::pso_trait::ParamValue;
use std::collections::HashMap;
mod pso;
mod pso_trait;
mod utils;
use pso::particle::Particle;
use pso::pso::PSO;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Experiment Settings
  let iterations = 1000;
  let out_directory = PathBuf::from("data/base_pso_test6");

  // Test particle_count vs. dimensions for CEC2017
  let cec17_dims = vec![2, 10, 20, 30, 50, 100];
  let particle_counts = vec![
    ParamValue::Int(2),
    ParamValue::Int(5),
    ParamValue::Int(10),
    ParamValue::Int(50),
    ParamValue::Int(100),
    ParamValue::Int(200),
    ParamValue::Int(500),
  ];
  let base_params: HashMap<String, ParamValue> = [
    ("w".to_owned(), ParamValue::Float(0.8)),
    ("phi_p".to_owned(), ParamValue::Float(1.0)),
    ("phi_g".to_owned(), ParamValue::Float(1.0)),
  ]
  .iter()
  .cloned()
  .collect();

  for func_num in 1..=5 {
    if func_num == 2 {
      continue;
    }
    utils::grid_search_dim::<Particle, PSO<Particle>>(
      iterations,
      Arc::new(move |d: usize| functions::cec17(func_num, d)),
      3,
      cec17_dims.clone(),
      ("particle_count".to_owned(), particle_counts.clone()),
      base_params.clone(),
      out_directory.clone(),
    )?;
  }

  Ok(())
}
