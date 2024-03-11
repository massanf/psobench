extern crate nalgebra as na;
extern crate rand;

mod defaultpso;
mod function;
mod particle;
mod problems;
mod pso;
mod utils;
use defaultpso::particle::DefaultParticle;
use defaultpso::pso::DefaultPSO;
use pso::PSOTrait;
use std::path::Path;
// use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Experiment Settings
  let problem = problems::f1();
  let dimensions = 30;
  let particle_count = 100;
  let iterations = 10000;

  // SPSO
  let mut pso: DefaultPSO<'_, DefaultParticle> = DefaultPSO::new("Default PSO", &problem, dimensions, particle_count);
  pso.run(iterations);
  pso.save_history(Path::new("data/PSO.json"))?;

  Ok(())
}
