extern crate nalgebra as na;
extern crate rand;

mod function;
mod particle_trait;
mod problems;
mod pso;
mod pso_trait;
mod utils;
use pso::particle::Particle;
use pso::pso::PSO;
use pso_trait::PSOTrait;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Experiment Settings
  let problem = problems::f1();
  let dimensions = 2;
  let particle_count = 5;
  let iterations = 100;

  // SPSO
  let mut pso: PSO<'_, Particle> = PSO::new("PSO", &problem, dimensions, particle_count);
  pso.run(iterations);
  pso.save_history(Path::new("data/PSO.json"))?;

  Ok(())
}
