extern crate nalgebra as na;
extern crate rand;
mod optimization_problem;
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
// use gsa::gsa::GSA;
// use gsa::particle::GSAParticle;
use pso::particle::Particle;
use pso::pso::PSO;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Experiment Settings
  let iterations = 1000;
  let base_params: HashMap<String, ParamValue> = [
    ("w".to_owned(), ParamValue::Float(0.8)),
    ("phi_p".to_owned(), ParamValue::Float(1.0)),
    ("phi_g".to_owned(), ParamValue::Float(1.0)),
    ("particle_count".to_owned(), ParamValue::Int(30)),
  ]
  .iter()
  .cloned()
  .collect();

  let mut gsa: PSO<Particle> = PSO::new(
    "PSO",
    functions::f1(30),
    base_params.clone(),
    PathBuf::from("data/test"),
  );
  gsa.run(iterations);
  gsa.save_summary()?;
  gsa.save_config()?;
  gsa.save_data()?;
  Ok(())
}
