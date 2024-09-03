extern crate nalgebra as na;
extern crate rand;
mod functions;
mod grid_search;
mod optimizers;
mod parameters;
mod particles;
mod problems;
mod utils;
use crate::optimizers::{gsa::Normalizer, traits::ParamValue};
#[allow(unused_imports)]
use optimizers::{gaussian::Gaussian, gsa::Gsa, pso::Pso};
#[allow(unused_imports)]
use particles::{
  gaussian::GaussianParticle,
  gsa::GsaParticle,
  pso::PsoParticle,
  traits::{Behavior, Edge},
};
#[allow(unused_imports)]
use strum::IntoEnumIterator;
#[allow(unused_imports)]
use ParamValue::Float as f;
#[allow(unused_imports)]
use ParamValue::Int as i;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let dims = [30];
  let iterations = 1000;
  let attempts = 10;
  let particle_count = 50;

  for dim in dims {
    utils::check_cec17::<GaussianParticle, Gaussian<GaussianParticle>>(
      "test",
      "gaussian",
      iterations,
      dim,
      attempts,
      vec![
        ("g0", f(1000.)),
        ("particle_count", i(particle_count)),
        ("gamma", f(0.8)),
        ("beta", f(0.4)),
        (
          "behavior",
          ParamValue::Behavior(Behavior {
            edge: Edge::Pass,
            vmax: false,
          }),
        ),
      ],
      true,
    )?;
  }
  Ok(())
}
