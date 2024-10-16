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
use optimizers::{gaussian::Gaussian, gsa::Gsa, mgsa::Mgsa, pso::Pso};
#[allow(unused_imports)]
use particles::{
  gaussian::GaussianParticle,
  gsa::GsaParticle,
  mgsa::MgsaParticle,
  pso::PsoParticle,
  traits::{Behavior, Edge},
};
use std::env;
#[allow(unused_imports)]
use strum::IntoEnumIterator;
#[allow(unused_imports)]
use ParamValue::Float as f;
#[allow(unused_imports)]
use ParamValue::Int as i;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args: Vec<String> = env::args().collect();

  match args[1].as_str() {
    "single" => single()?,
    "cec" => cec()?,
    "grid" => grid()?,
    _ => panic!("Unknown argument: {}. Please use fn1, fn2, or fn3", args[1]),
  }

  Ok(())
}

fn single() -> Result<(), Box<dyn std::error::Error>> {
  let dims = [30];
  let iterations = 1000;
  let attempts = 1;
  let particle_count = 50;

  for dim in dims {
    let problem = problems::cec17(5, dim);
    // let problem = problems::sphere_100(dim);
    utils::check_problem::<MgsaParticle, Mgsa<MgsaParticle>>(
      "test",
      "mgsa_test",
      iterations,
      dim,
      attempts,
      vec![
        ("particle_count", i(particle_count)),
        ("w", f(0.5)),
        ("alpha", f(5.)),
        ("g0", f(1000.)),
        ("theta", f(1.)),
        ("tiled", ParamValue::Tiled(false)),
        ("normalizer", ParamValue::Normalizer(Normalizer::MinMax)),
        (
          "behavior",
          ParamValue::Behavior(Behavior {
            edge: Edge::Pass,
            vmax: false,
          }),
        ),
      ],
      problem.clone(),
      true,
    )?;
    utils::check_problem::<GsaParticle, Gsa<GsaParticle>>(
      "test",
      "gsa_test",
      iterations,
      dim,
      attempts,
      vec![
        ("particle_count", i(particle_count)),
        ("w", f(0.5)),
        ("alpha", f(5.)),
        ("g0", f(1000.)),
        ("tiled", ParamValue::Tiled(false)),
        ("normalizer", ParamValue::Normalizer(Normalizer::MinMax)),
        (
          "behavior",
          ParamValue::Behavior(Behavior {
            edge: Edge::Pass,
            vmax: false,
          }),
        ),
      ],
      problem.clone(),
      true,
    )?;
  }
  Ok(())
}

fn cec() -> Result<(), Box<dyn std::error::Error>> {
  let dims = [30];
  let iterations = 1000;
  let attempts = 1;
  let particle_count = 50;

  for dim in dims {
    utils::check_cec17::<MgsaParticle, Mgsa<MgsaParticle>>(
      "test",
      "mgsa_test",
      iterations,
      dim,
      attempts,
      vec![
        ("particle_count", i(particle_count)),
        ("alpha", f(5.)),
        ("g0", f(100.)),
        ("theta", f(1.)),
        ("tiled", ParamValue::Tiled(false)),
        ("normalizer", ParamValue::Normalizer(Normalizer::MinMax)),
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
    utils::check_cec17::<GsaParticle, Gsa<GsaParticle>>(
      "test",
      "gsa_test",
      iterations,
      dim,
      attempts,
      vec![
        ("particle_count", i(particle_count)),
        ("alpha", f(5.)),
        ("g0", f(100.)),
        ("tiled", ParamValue::Tiled(false)),
        ("normalizer", ParamValue::Normalizer(Normalizer::MinMax)),
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

fn grid() -> Result<(), Box<dyn std::error::Error>> {
  let dims = [30];
  let iterations = 1000;
  let attempts = 10;
  let particle_count = 50;

  for dim in dims {
    utils::run_grid_searches::<MgsaParticle, Mgsa<MgsaParticle>>(
      "mgsa_test",
      attempts,
      iterations,
      dim,
      (
        "alpha".to_owned(),
        vec![f(1.0), f(2.0), f(5.0), f(10.0), f(20.0), f(50.0), f(100.0)],
      ),
      (
        "g0".to_owned(),
        vec![
          f(2.0),
          f(5.0),
          f(10.0),
          f(20.0),
          f(50.0),
          f(100.0),
          f(200.0),
          f(500.0),
          f(1000.0),
          f(2000.0),
          f(5000.0),
          f(10000.0),
        ],
      ),
      vec![
        ("particle_count", i(particle_count)),
        ("tiled", ParamValue::Tiled(false)),
        ("theta", f(1.0)),
        ("normalizer", ParamValue::Normalizer(Normalizer::MinMax)),
        (
          "behavior",
          ParamValue::Behavior(Behavior {
            edge: Edge::Pass,
            vmax: false,
          }),
        ),
      ],
    )?;
  }
  Ok(())
}
