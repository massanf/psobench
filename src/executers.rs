extern crate nalgebra as na;
extern crate rand;
use crate::optimizers::{gsa::Gsa, pso::Pso, tiled_gsa::TiledGSA};
use crate::parameters;
use crate::particles::{
  gsa_particle::GsaParticle, igsa_particle::IgsaParticle, pso_particle::PsoParticle, traits::Behavior,
};
use crate::utils;
use std::path::PathBuf;

#[allow(dead_code)]
pub fn pso_cec17(
  iterations: usize,
  dim: usize,
  attempts: usize,
  behavior: Behavior,
) -> Result<(), Box<dyn std::error::Error>> {
  utils::check_cec17::<PsoParticle, Pso<PsoParticle>>(
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
  utils::check_cec17::<GsaParticle, Gsa<GsaParticle>>(
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
pub fn igsa_cec17(
  iterations: usize,
  dim: usize,
  attempts: usize,
  behavior: Behavior,
) -> Result<(), Box<dyn std::error::Error>> {
  utils::check_cec17::<IgsaParticle, Gsa<IgsaParticle>>(
    "IGSA".to_owned(),
    iterations,
    dim,
    parameters::GSA_PARAMS.clone(),
    attempts,
    PathBuf::from(format!("data/test/{}/igsa", dim)),
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
  utils::check_cec17::<GsaParticle, TiledGSA<GsaParticle>>(
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
  utils::run_grid_searches::<PsoParticle, Pso<PsoParticle>>(
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
  utils::run_grid_searches::<GsaParticle, Gsa<GsaParticle>>(
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

#[allow(dead_code)]
pub fn grid_search_igsa(
  iterations: usize,
  dim: usize,
  attempts: usize,
  behavior: Behavior,
) -> Result<(), Box<dyn std::error::Error>> {
  utils::run_grid_searches::<IgsaParticle, Gsa<IgsaParticle>>(
    "IGSA".to_owned(),
    attempts,
    iterations,
    parameters::GSA_G0_OPTIONS.clone(),
    parameters::GSA_ALPHA_OPTIONS.clone(),
    parameters::GSA_BASE_PARAMS.clone(),
    dim,
    PathBuf::from(format!("data/grid_search/{}/igsa", dim)),
    behavior,
  )?;
  Ok(())
}
