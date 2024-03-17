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

  // PSO
  let mut pso: PSO<'_, Particle> = PSO::new(
    "PSO",
    &problem,
    particle_count,
    [
      ("w".to_owned(), 0.8),
      ("phi_p".to_owned(), 1.),
      ("phi_g".to_owned(), 2.),
    ]
    .iter()
    .cloned()
    .collect(),
  );
  pso.run(iterations);
  pso.save_history(Path::new("data/PSO.json"))?;
  pso.save_settings(Path::new("data/PSO_config.json"))?;

  Ok(())
}
