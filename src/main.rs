extern crate nalgebra as na;
extern crate rand;
mod functions;
mod grid_search;
mod optimizers;
mod parameters;
mod particles;
mod problems;
mod utils;
use crate::optimizers::traits::ParamValue;
#[allow(unused_imports)]
use optimizers::{gsa::Gsa, pso::Pso};
#[allow(unused_imports)]
use particles::{
  gsa_particle::GsaParticle,
  igsa_particle::IgsaParticle,
  pso_particle::PsoParticle,
  traits::{Behavior, Edge},
};
use ParamValue::Float as f;
use ParamValue::Int as i;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // let dims = [10, 30, 50, 100];
  let dim = 30;
  let iterations = 1000;
  let attempts = 10;
  let name = "gsa";

  let behavior = Behavior {
    edge: Edge::Cycle,
    vmax: true,
  };

  utils::check_cec17::<IgsaParticle, Gsa<IgsaParticle>>(
    name,
    iterations,
    dim,
    attempts,
    behavior,
    vec![("g0", f(5000.0)), ("alpha", f(5.0)), ("particle_count", i(30))],
    format!("data/test/{}/{}", dim, name),
  )?;

  utils::run_grid_searches::<PsoParticle, Pso<PsoParticle>>(
    "PSO".to_owned(),
    attempts,
    iterations,
    dim,
    parameters::PSO_PHI_P_OPTIONS.clone(),
    parameters::PSO_PHI_G_OPTIONS.clone(),
    vec![("w", f(0.8)), ("particle_count", i(30))],
    format!("data/grid_search/{}/pso", dim),
    behavior,
  )?;

  Ok(())
}
