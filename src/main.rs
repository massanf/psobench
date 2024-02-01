extern crate nalgebra as na;
extern crate rand;

use std::process::Command;
mod awparticle;
mod defaultparticle;
mod defaultpso;
mod function;
mod particle;
mod problems;
mod pso;
mod utils;
use awparticle::AWParticle;
use defaultparticle::DefaultParticle;
use defaultpso::DefaultPSO;
use pso::PSOTrait;
use std::path::Path;

fn main() {
  // Experimental Settings
  let problem = problems::f2();
  let dimensions = 30;
  let particle_count = 100;
  let iterations = 10000;

  // SPSO
  let mut pso: DefaultPSO<'_, DefaultParticle> = DefaultPSO::new(&problem, dimensions, particle_count);
  pso.run(iterations);
  let _ = pso.export_global_best_progress(Path::new("data/PSO.csv"));

  // AWPSO
  let mut awpso: DefaultPSO<AWParticle> = DefaultPSO::new(&problem, dimensions, particle_count);
  awpso.run(iterations);
  let _ = awpso.export_global_best_progress(Path::new("data/AWPSO.csv"));

  // Generate progress graph
  let output = Command::new("python").arg("visualizer.py").output().expect("Failed to execute command");
  if output.status.success() {
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("{}", stdout);
  } else {
    let stderr = String::from_utf8_lossy(&output.stderr);
    eprintln!("Visualizer generated an error: {}", stderr);
  }
}
