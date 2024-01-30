extern crate nalgebra as na;
extern crate rand;

use nalgebra::DVector;
use std::path::Path;
mod awparticle;
mod defaultparticle;
mod grapher;
mod particle;
mod pso;
mod utils;

use awparticle::AWParticle;
use defaultparticle::DefaultParticle;
use pso::PSO;

fn main() {
  let dimensions = 30;
  let particle_count = 100;
  fn f(x: &DVector<f64>) -> f64 {
    x.iter().map(|&x| x * x).sum()
  }

  let mut pso: PSO<DefaultParticle> = pso::PSO::new(f, dimensions, particle_count);
  pso.run(100);
  println!("PSO: {}", f(&pso.global_best_pos()));
  let _ = grapher::generate_progress_graph(Path::new("graphs/pso.png"), pso.data());

  let mut awpso: PSO<AWParticle> = pso::PSO::new(f, dimensions, particle_count);
  awpso.run(100);
  println!("AWPSO: {}", f(&awpso.global_best_pos()));
  let _ = grapher::generate_progress_graph(Path::new("graphs/awpso.png"), awpso.data());
}
