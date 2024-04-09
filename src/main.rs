extern crate nalgebra as na;
extern crate rand;
mod executers;
mod fdo;
mod functions;
mod grid_search;
mod gsa;
mod parameters;
mod particle_trait;
mod problem;
mod pso;
mod pso_trait;
mod utils;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let dim = 30;
  let iterations = 1000;
  let attempts = 10;

  executers::pso_cec17(iterations, dim, attempts)?;
  // executers::gsa_cec17(iterations, dim, attempts)?;
  // executers::igsa_cec17(iterations, dim, attempts)?;

  // executers::grid_search_pso(iterations, dim, attempts)?;
  // executers::grid_search_gsa(iterations, dim, attempts)?;
  // executers::grid_search_igsa(iterations, dim, attempts)?;

  Ok(())
}
