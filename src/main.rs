extern crate nalgebra as na;
extern crate rand;
mod optimization_problem;
use std::path::PathBuf;
mod functions;
mod particle_trait;
mod pso;
mod pso_trait;
mod utils;
use pso::particle::Particle;
use pso::pso::PSO;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Problem Settings
  let problem = functions::cec17(1, 30);

  // Experiment Settings
  let particle_count = 30;
  let iterations = 1500;

  // PSO
  utils::grid_search::<'_, Particle, PSO<'_, Particle>>(
    particle_count,
    iterations,
    &problem,
    1,
    vec![("w".to_owned(), (-2.0, 2.0)), ("phi_g".to_owned(), (-4.0, 4.0))],
    [("phi_p".to_owned(), 1.0)].iter().cloned().collect(),
    PathBuf::from("data/test"),
  )?;

  Ok(())
}
