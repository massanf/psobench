extern crate nalgebra as na;
extern crate rand;
mod executers;
mod functions;
mod grid_search;
mod optimizers;
mod parameters;
mod particles;
mod problems;
mod utils;
use particles::traits::{Behavior, Edge};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let dim = 30;
  let iterations = 1000;
  let attempts = 10;

  let behavior = Behavior {
    edge: Edge::Pass,
    vmax: false,
  };

  executers::gsa_cec17(iterations, dim, attempts, behavior)?;
  // executers::gsa_cec17(iterations, dim, attempts, behavior)?;

  // executers::grid_search_pso(iterations, dim, attempts)?;
  // executers::grid_search_gsa(iterations, dim, attempts, behavior)?;
  // executers::grid_search_igsa(iterations, dim, attempts)?;

  Ok(())
}
