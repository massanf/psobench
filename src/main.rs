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
  let dims = [10];
  let iterations = 100;
  let attempts = 1;
  let particle_count = 100;

  for dim in dims {
    utils::check_problem::<PsoParticle, Pso<PsoParticle>>(
      "test",
      "pso_test",
      iterations,
      dim,
      attempts,
      vec![
        ("particle_count", i(particle_count)),
        ("w", f(0.5)),
        ("phi_p", f(2.)),
        ("phi_g", f(2.)),
        (
          "behavior",
          ParamValue::Behavior(Behavior {
            edge: Edge::Pass,
            vmax: false,
          }),
        ),
      ],
      problems::cec17(1, dim),
      true,
    )?;

    utils::check_problem::<GaussianParticle, Gaussian<GaussianParticle>>(
      "test",
      "gaussian_test",
      iterations,
      dim,
      attempts,
      vec![
        ("particle_count", i(particle_count)),
        ("gamma", f(0.8)),
        ("beta", f(0.4)),
        ("scale", f(1.)),
        (
          "behavior",
          ParamValue::Behavior(Behavior {
            edge: Edge::Pass,
            vmax: false,
          }),
        ),
      ],
      problems::cec17(1, dim),
      true,
    )?;
    // utils::check_cec17::<GaussianParticle, Gaussian<GaussianParticle>>(
    //   "test",
    //   "gaussian_test",
    //   iterations,
    //   dim,
    //   attempts,
    //   vec![
    //     ("particle_count", i(particle_count)),
    //     ("gamma", f(0.8)),
    //     ("beta", f(0.4)),
    //     ("scale", f(1.)),
    //     (
    //       "behavior",
    //       ParamValue::Behavior(Behavior {
    //         edge: Edge::Pass,
    //         vmax: false,
    //       }),
    //     ),
    //   ],
    //   true,
    // )?;
  }
  Ok(())
}
