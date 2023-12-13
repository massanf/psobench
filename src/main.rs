extern crate nalgebra as na;
extern crate rand;

mod grapher;
mod particle;
mod pso;
mod utils;

use particle::Particle;
use pso::PSO;

fn main() {
  let mut pso: PSO<Particle> = pso::PSO::new(100, 300);
  // `PSO` has to be initialized after being created.
  // TODO: Fix this.
  pso.init();

  pso.run(10000);
}
