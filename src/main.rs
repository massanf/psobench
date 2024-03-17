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
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Problem Settings
  let problem = problems::f3(30);

  // Experiment Settings
  let particle_count = 30;
  let iterations = 1500;

  // PSO
  let steps: usize = 20;
  let attempts: usize = 20;
  for p in 0..steps {
    for g in 0..steps {
      let p = p as f64 / steps as f64 * 5.0;
      let g = g as f64 / steps as f64 * 5.0;
      for attempt in 0..attempts {
        let mut pso: PSO<'_, Particle> = PSO::new(
          "PSO",
          &problem,
          particle_count,
          [("w".to_owned(), 0.8), ("phi_p".to_owned(), p), ("phi_g".to_owned(), g)].iter().cloned().collect(),
          PathBuf::from(format!("data/PSO_grid/p{}_g{}/{}", p, g, attempt)),
        );
        pso.run(iterations);
        pso.save_summary()?;
        pso.save_config()?;
      }
    }
  }

  Ok(())
}
