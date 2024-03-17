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
  // Problem Settings
  let problem = problems::f3(30);

  // Experiment Settings
  let particle_count = 30;
  let iterations = 3000;

  // SPSO
  let mut pso: PSO<'_, Particle> = PSO::new("PSO", &problem, particle_count);
  pso.run(iterations);
  pso.save_history(Path::new("data/PSO.json"))?;
  pso.save_settings(Path::new("data/PSO_settings.json"))?;

  Ok(())
}
