extern crate nalgebra as na;
extern crate rand;

use nalgebra::DVector;
mod awparticle;
mod grapher;
mod particle;
mod pso;
mod utils;

use awparticle::AWParticle;
use particle::Particle;
use pso::PSO;

fn main() {
  let dimensions = 30;
  let particle_count = 10;
  fn f(x: &DVector<f64>) -> f64 {
    x.iter().map(|&x| x * x).sum()
  }

  let mut pso: PSO<Particle> = pso::PSO::new(f, dimensions, particle_count);
  println!("PSO: {}", pso.run(10000));

  let mut awpso: PSO<AWParticle> = pso::PSO::new(f, dimensions, particle_count);
  println!("AWPSO: {}", awpso.run(10000));
}
