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
use optimizers::{gsa::Gsa, mgsa::Mgsa, pso::Pso, rgsa::Rgsa};
#[allow(unused_imports)]
use particles::{
  gsa::GsaParticle,
  mgsa::MgsaParticle,
  pso::PsoParticle,
  rgsa::RgsaParticle,
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

  let elite = true;
  let g0 = 1.;
  let alpha = 5.;
  let gamma = 1.0; // fix
  let theta = 0.0; // fix
  let sigma = 10.;
  let edge = Edge::Pass;
  let dims = vec![10];

  match args[1].as_str() {
    "single" => single(dims, g0, alpha, gamma, theta, elite, sigma, edge)?,
    "cec" => cec(dims, g0, alpha, gamma, theta, elite, sigma, edge)?,
    "grid" => grid(dims, g0, alpha, gamma, theta, elite, sigma, edge)?,
    _ => panic!("Unknown argument: {}. Please use fn1, fn2, or fn3", args[1]),
  }

  Ok(())
}

#[allow(clippy::too_many_arguments)]
fn single(
  dims: Vec<usize>,
  _g0: f64,
  _alpha: f64,
  _gamma: f64,
  _theta: f64,
  _elite: bool,
  _sigma: f64,
  _edge: Edge,
) -> Result<(), Box<dyn std::error::Error>> {
  let iterations = 1000;
  let attempts = 10;
  let particle_count = 50;
  println!("aasdf");

  for dim in dims {
    let problem = problems::cec17(1, dim);

    utils::check_problem::<GsaParticle, Gsa<GsaParticle>>(
      "test",
      "ogsa_test",
      iterations,
      dim,
      attempts,
      vec![
        ("particle_count", i(particle_count)),
        ("alpha", f(5.)),
        ("g0", f(1000.)),
        ("tiled", ParamValue::Bool(false)),
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

#[allow(clippy::too_many_arguments)]
fn cec(
  dims: Vec<usize>,
  _g0: f64,
  _alpha: f64,
  _gamma: f64,
  _theta: f64,
  _elite: bool,
  _sigma: f64,
  _edge: Edge,
) -> Result<(), Box<dyn std::error::Error>> {
  let iterations = 1000;
  let attempts = 10;
  let particle_count = 50;

  for dim in dims {
    utils::check_cec17::<GsaParticle, Gsa<GsaParticle>>(
      "gachi_test",
      "mgsa_test_50_elite",
      iterations,
      dim,
      attempts,
      vec![
        ("particle_count", i(particle_count)),
        ("alpha", f(5.)),
        ("g0", f(1000.)),
        ("tiled", ParamValue::Bool(false)),
        ("manual_k", f(50.)),
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

#[allow(clippy::too_many_arguments)]
fn grid(
  dims: Vec<usize>,
  _g0: f64,
  alpha: f64,
  gamma: f64,
  theta: f64,
  elite: bool,
  _sigma: f64,
  edge: Edge,
) -> Result<(), Box<dyn std::error::Error>> {
  let iterations = 10000;
  let attempts = 1;
  let particle_count = 50;

  for dim in dims {
    utils::run_grid_searches::<MgsaParticle, Mgsa<MgsaParticle>>(
      "mgsa_test",
      attempts,
      iterations,
      dim,
      (
        "sigma".to_owned(),
        vec![
          f(1000.0),
          f(5000.0),
          f(10000.0),
          f(50000.0),
          f(100000.0),
          f(500000.0),
          f(1000000.0),
        ],
      ),
      (
        "g0".to_owned(),
        vec![f(0.01), f(0.05), f(0.1), f(0.5), f(1.0), f(5.0), f(10.0)],
      ),
      vec![
        ("particle_count", i(particle_count)),
        ("gamma", f(gamma)),
        ("theta", f(theta)),
        ("alpha", f(alpha)),
        ("elite", ParamValue::Bool(elite)),
        ("tiled", ParamValue::Bool(false)),
        ("normalizer", ParamValue::Normalizer(Normalizer::MinMax)),
        ("behavior", ParamValue::Behavior(Behavior { edge, vmax: false })),
      ],
    )?;
  }
  Ok(())
}
