extern crate nalgebra as na;
extern crate rand;

mod grapher;
mod particle;
mod pso;
mod utils;

fn main() {
    let mut pso = pso::PSO::new(2, 30);
    pso.init();
    pso.run(10);
}
