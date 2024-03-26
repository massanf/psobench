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
use gsa::gsa::GSA;
use gsa::particle::GSAParticle;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Experiment Settings
  let iterations = 1000;
  let params: HashMap<String, ParamValue> = [
    ("particle_count".to_owned(), ParamValue::Int(30)),
    ("g0".to_owned(), ParamValue::Float(100.)),
    ("alpha".to_owned(), ParamValue::Float(20.)),
    ("epsilon".to_owned(), ParamValue::Float(0.)),
  ]
  .iter()
  .cloned()
  .collect();

  let mut gsa: GSA<GSAParticle> = GSA::new("GSA", functions::f1(5), params.clone(), PathBuf::from("data/gsa"));
  gsa.run(iterations);
  gsa.save_summary()?;
  gsa.save_config(&params)?;
  gsa.save_data()?;
  Ok(())
}
