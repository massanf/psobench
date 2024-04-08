extern crate nalgebra as na;
extern crate rand;
mod parameters;
mod problem;
use crate::pso_trait::DataExporter;
use std::path::PathBuf;
mod functions;
mod particle_trait;
use crate::pso_trait::ParamValue;
mod fdo;
mod grid_search;
mod gsa;
mod pso;
mod pso_trait;
use crate::gsa::tiled_gsa::TiledGSA;
use crate::pso_trait::ParticleOptimizer;
use fdo::particle::FDOParticle;
use gsa::gsa::GSA;
use gsa::gsa_particle::GSAParticle;
use gsa::tiled_gsa_particle::TiledGSAParticle;
use pso::particle::PSOParticle;
use pso::pso::PSO;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let dim = 30;
  let iterations = 1000;
  let attempts = 30;

  utils::check_cec17::<PSOParticle, PSO<PSOParticle>>(
    "PSO".to_owned(),
    iterations,
    dim,
    parameters::PSO_PARAMS.clone(),
    attempts,
    PathBuf::from(format!("data/test/{}/pso", dim)),
  )?;

  utils::check_cec17::<GSAParticle, GSA<GSAParticle>>(
    "GSA".to_owned(),
    iterations,
    dim,
    parameters::GSA_PARAMS.clone(),
    attempts,
    PathBuf::from(format!("data/test/{}/gsa", dim)),
  )?;

  utils::check_cec17::<TiledGSAParticle, TiledGSA<TiledGSAParticle>>(
    "TiledGSA".to_owned(),
    iterations,
    dim,
    parameters::TILED_GSA_PARAMS.clone(),
    attempts,
    PathBuf::from(format!("data/test/{}/tiled_gsa", dim)),
  )?;

  utils::run_grid_searches::<PSOParticle, PSO<PSOParticle>>(
    "PSO".to_owned(),
    attempts,
    parameters::PSO_PHI_P_OPTIONS.clone(),
    parameters::PSO_PHI_G_OPTIONS.clone(),
    parameters::PSO_BASE_PARAMS.clone(),
    dim,
    PathBuf::from(format!("data/grid_search/{}/pso", dim)),
  )?;

  utils::run_grid_searches::<GSAParticle, GSA<GSAParticle>>(
    "GSA".to_owned(),
    attempts,
    parameters::GSA_G0_OPTIONS.clone(),
    parameters::GSA_ALPHA_OPTIONS.clone(),
    parameters::GSA_BASE_PARAMS.clone(),
    dim,
    PathBuf::from(format!("data/grid_search/{}/gsa", dim)),
  )?;

  utils::run_grid_searches::<TiledGSAParticle, TiledGSA<TiledGSAParticle>>(
    "GSA".to_owned(),
    attempts,
    parameters::GSA_G0_OPTIONS.clone(),
    parameters::GSA_ALPHA_OPTIONS.clone(),
    parameters::GSA_BASE_PARAMS.clone(),
    dim,
    PathBuf::from(format!("data/grid_search/{}/tiled_gsa", dim)),
  )?;

  Ok(())
}
