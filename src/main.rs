extern crate nalgebra as na;
extern crate rand;

use std::path::Path;
mod awparticle;
mod defaultparticle;
mod function;
mod grapher;
mod particle;
mod pso;
mod utils;

use awparticle::AWParticle;
use defaultparticle::DefaultParticle;
use function::OptimizationProblem;
use pso::PSO;

fn main() {
  // Experimental Settings
  let f1 = OptimizationProblem::new(|x| x.iter().map(|&x| x * x).sum(), (-1., 1.));

  let dimensions = 30;
  let particle_count = 100;
  let iterations = 100;

  // SPSO
  let mut pso: PSO<DefaultParticle> = pso::PSO::new(&f1, dimensions, particle_count);
  pso.run(iterations);
  println!("PSO: {}", f1.f(&pso.global_best_pos()));
  let _ = grapher::progress_graph(Path::new("graphs/pso.png"), pso.data());

  // AWPSO
  let mut awpso: PSO<AWParticle> = pso::PSO::new(&f1, dimensions, particle_count);
  awpso.run(iterations);
  println!("AWPSO: {}", f1.f(&awpso.global_best_pos()));
  let _ = grapher::progress_graph(Path::new("graphs/awpso.png"), awpso.data());
}
