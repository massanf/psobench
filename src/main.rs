extern crate nalgebra as na;
extern crate rand;

use nalgebra::DVector;
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
  println!("PSO: {}", pso.run(10000));

  let mut awpso: PSO<AWParticle> = pso::PSO::new(f, dimensions, particle_count);
  println!("AWPSO: {}", awpso.run(10000));
}
