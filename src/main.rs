extern crate nalgebra as na;
extern crate rand;

mod awpso;
mod defaultpso;
mod function;
mod particle;
mod pppso;
mod problems;
mod pso;
mod utils;
use awpso::particle::AWParticle;
use defaultpso::particle::DefaultParticle;
use defaultpso::pso::DefaultPSO;
use pppso::pso::PPPSO;
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

  // AWPSO
  let mut awpso: DefaultPSO<AWParticle> = DefaultPSO::new("AWPSO", &problem, dimensions, particle_count);
  awpso.run(iterations);
  awpso.save_history(Path::new("data/AWPSO.json"))?;

  // PPPSO
  let mut pppso: PPPSO<DefaultParticle> = PPPSO::new("PPPSO", &problem, dimensions, particle_count);
  pppso.run(iterations);
  pppso.save_history(Path::new("data/PPSO.json"))?;

  Ok(())
}
