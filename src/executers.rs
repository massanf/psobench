extern crate nalgebra as na;
extern crate rand;
use crate::optimizers::{gsa::Gsa, pso::Pso, tiled_gsa::TiledGSA};
use crate::parameters;
use crate::particles::{gsa_particle::GSAParticle, pso_particle::PSOParticle, traits::Behavior};
use crate::utils;
use std::path::PathBuf;

#[allow(dead_code)]
pub fn pso_cec17(
  iterations: usize,
  dim: usize,
  attempts: usize,
  behavior: Behavior,
) -> Result<(), Box<dyn std::error::Error>> {
  utils::check_cec17::<PSOParticle, Pso<PSOParticle>>(
    "PSO".to_owned(),
    iterations,
    dim,
    parameters::PSO_PARAMS.clone(),
    attempts,
    PathBuf::from(format!("data/test/{}/pso", dim)),
    behavior,
  )?;
  Ok(())
}

#[allow(dead_code)]
pub fn gsa_cec17(
  iterations: usize,
  dim: usize,
  attempts: usize,
  behavior: Behavior,
) -> Result<(), Box<dyn std::error::Error>> {
  utils::check_cec17::<GSAParticle, Gsa<GSAParticle>>(
    "GSA".to_owned(),
    iterations,
    dim,
    parameters::GSA_PARAMS.clone(),
    attempts,
    PathBuf::from(format!("data/test/{}/gsa", dim)),
    behavior,
  )?;
  Ok(())
}

#[allow(dead_code)]
pub fn tiled_gsa_cec17(
  iterations: usize,
  dim: usize,
  attempts: usize,
  behavior: Behavior,
) -> Result<(), Box<dyn std::error::Error>> {
  utils::check_cec17::<GSAParticle, TiledGSA<GSAParticle>>(
    "TiledGSA".to_owned(),
    iterations,
    dim,
    parameters::TILED_GSA_PARAMS.clone(),
    attempts,
    PathBuf::from(format!("data/test/{}/tiled_gsa", dim)),
    behavior,
  )?;
  Ok(())
}

#[allow(dead_code)]
pub fn grid_search_pso(
  iterations: usize,
  dim: usize,
  attempts: usize,
  behavior: Behavior,
) -> Result<(), Box<dyn std::error::Error>> {
  utils::run_grid_searches::<PSOParticle, Pso<PSOParticle>>(
    "PSO".to_owned(),
    attempts,
    iterations,
    parameters::PSO_PHI_P_OPTIONS.clone(),
    parameters::PSO_PHI_G_OPTIONS.clone(),
    parameters::PSO_BASE_PARAMS.clone(),
    dim,
    PathBuf::from(format!("data/grid_search/{}/pso", dim)),
    behavior,
  )?;
  Ok(())
}

#[allow(dead_code)]
pub fn grid_search_gsa(
  iterations: usize,
  dim: usize,
  attempts: usize,
  behavior: Behavior,
) -> Result<(), Box<dyn std::error::Error>> {
  utils::run_grid_searches::<GSAParticle, Gsa<GSAParticle>>(
    "GSA".to_owned(),
    attempts,
    iterations,
    parameters::GSA_G0_OPTIONS.clone(),
    parameters::GSA_ALPHA_OPTIONS.clone(),
    parameters::GSA_BASE_PARAMS.clone(),
    dim,
    PathBuf::from(format!("data/grid_search/{}/gsa", dim)),
    behavior,
  )?;
  Ok(())
}
