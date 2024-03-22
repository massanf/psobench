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
  let mut problem_set = Vec::new();
  for i in 1..=5 {
    if i == 2 {
      continue;
    }
    problem_set.push(functions::cec17(i, 30));
  }

  // Experiment Settings
  let particle_count = 30;
  let iterations = 1000;

  // PSO
  utils::grid_search::<'_, Particle, PSO<'_, Particle>>(
    particle_count,
    iterations,
    &problem_set,
    10,
    vec![("phi_p".to_owned(), (-4.0, 4.0)), ("phi_g".to_owned(), (-4.0, 4.0))],
    [("w".to_owned(), 0.8)].iter().cloned().collect(),
    PathBuf::from("data/test3"),
  )?;

  Ok(())
}
