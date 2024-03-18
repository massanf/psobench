extern crate nalgebra as na;
extern crate rand;
mod function;
use std::path::PathBuf;
mod particle_trait;
mod problems;
mod pso;
mod pso_trait;
mod utils;
use pso::particle::Particle;
use pso::pso::PSO;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Problem Settings
  let problem = problems::f2(30);

  // Experiment Settings
  let particle_count = 30;
  let iterations = 1500;

  // PSO
  utils::grid_search::<'_, Particle, PSO<'_, Particle>>(
    particle_count,
    iterations,
    &problem,
    1,
    vec![("phi_p".to_owned(), (0.0, 5.0)), ("phi_g".to_owned(), (0.0, 5.0))],
    [("w".to_owned(), 0.8)].iter().cloned().collect(),
    PathBuf::from("data/test"),
  )?;

  Ok(())
}
