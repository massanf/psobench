extern crate nalgebra as na;
extern crate rand;

use nalgebra::DVector;
mod grapher;
mod particle;
mod pso;
mod utils;

use particle::Particle;
use pso::PSO;

fn main() {
  fn f(x: &DVector<f64>) -> f64 {
    x.iter().map(|&x| x * x).sum()
  }

  let mut pso: PSO<Particle> = pso::PSO::new(f, 100, 300);

  pso.run(10000);
}
