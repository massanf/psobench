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
use optimizers::{
  //gaussian::Gaussian,
  gsa::Gsa,
  mgsa::Mgsa,
  pso::Pso,
  rgsa::Rgsa,
};
#[allow(unused_imports)]
use particles::{
  // gaussian::GaussianParticle,
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
      println!("{}", dim);
    // let problem = problems::sphere_100(dim);
    let mut func_nums = Vec::new();
    for func_num in 0..=30 {
      if func_num == 2 {
        continue;
      }
      if func_num != 1 && func_num != 3 {
        continue;
      }
      func_nums.push(func_num);
    }
    println!("{:?}", func_nums);

    for p in func_nums {
      println!("problem: {}", p);
      let problem = problems::cec17(p, dim);

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
      // utils::check_problem::<PsoParticle, Pso<PsoParticle>>(
      //   "test",
      //   "pso_test",
      //   iterations,
      //   dim,
      //   attempts,
      //   vec![
      //     ("particle_count", i(particle_count)),
      //     ("w", f(0.5)),
      //     ("phi_p", f(2.)),
      //     ("phi_g", f(2.)),
      //     (
      //       "behavior",
      //       ParamValue::Behavior(Behavior {
      //         edge: Edge::Pass,
      //         vmax: false,
      //       }),
      //     ),
      //   ],
      //   problem.clone(),
      //   true,
      // )?;

      for k in [5, 10, 25, 50, 100] {
        // for k in [25] {
        // utils::check_problem::<MgsaParticle, Mgsa<MgsaParticle>>(
        //   "test",
        //   "mgsa_test",
        //   iterations,
        //   dim,
        //   attempts,
        //   vec![
        //     ("particle_count", i(particle_count)),
        //     ("g0", f(g0)),
        //     ("theta", f(theta)),
        //     ("alpha", f(alpha)),
        //     ("gamma", f(gamma)),
        //     ("sigma", f(sigma)),
        //     ("tiled", ParamValue::Bool(false)),
        //     ("elite", ParamValue::Bool(elite)),
        //     ("normalizer", ParamValue::Normalizer(Normalizer::Sigmoid2)),
        //     (
        //       "behavior",
        //       ParamValue::Behavior(Behavior {
        //         edge: edge,
        //         vmax: false,
        //       }),
        //     ),
        //   ],
        //   problem.clone(),
        //   true,
        // )?;
        // println!("=== {} ===", format!("ogsa_test_{:03}_elite", k).as_str());
        // utils::check_problem::<GsaParticle, Gsa<GsaParticle>>(
        //   "test2",
        //   // "ogsa_test_100_elite",
        //   format!("ogsa_test_{:03}_elite", k).as_str(),
        //   iterations,
        //   dim,
        //   attempts,
        //   vec![
        //     ("particle_count", i(particle_count)),
        //     ("alpha", f(5.)),
        //     ("g0", f(1000.)),
        //     ("tiled", ParamValue::Bool(false)),
        //     ("normalizer", ParamValue::Normalizer(Normalizer::MinMax)),
        //     ("manual_k", f(k as f64)),
        //     (
        //       "behavior",
        //       ParamValue::Behavior(Behavior {
        //         edge: Edge::Pass,
        //         vmax: false,
        //       }),
        //     ),
        //   ],
        //   problem.clone(),
        //   true,
        // )?;
        // utils::check_problem::<RgsaParticle, Rgsa<RgsaParticle>>(
        //   "test",
        //   "rgsa_test",
        //   iterations,
        //   dim,
        //   attempts,
        //   vec![
        //     ("particle_count", i(particle_count)),
        //     ("alpha", f(5.)),
        //     ("g0", f(1000.)),
        //     ("tiled", ParamValue::Bool(false)),
        //     ("normalizer", ParamValue::Normalizer(Normalizer::MinMax)),
        //     (
        //       "behavior",
        //       ParamValue::Behavior(Behavior {
        //         edge: Edge::Pass,
        //         vmax: false,
        //       }),
        //     ),
        //   ],
        //   problem.clone(),
        //   true,
        // )?;
        // utils::check_problem::<PsoParticle, Pso<PsoParticle>>(
        //   "test",
        //   "pso_test",
        //   iterations,
        //   dim,
        //   attempts,
        //   vec![
        //     ("particle_count", i(particle_count)),
        //     ("w", f(0.5)),
        //     ("phi_p", f(2.)),
        //     ("phi_g", f(2.)),
        //     (
        //       "behavior",
        //       ParamValue::Behavior(Behavior {
        //         edge: Edge::Pass,
        //         vmax: false,
        //       }),
        //     ),
        //   ],
        //   problem.clone(),
        //   true,
        // )?;
        //
      }
    }
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
    // utils::check_cec17::<MgsaParticle, Mgsa<MgsaParticle>>(
    //   "test",
    //   "mgsa_test",
    //   iterations,
    //   dim,
    //   attempts,
    //   vec![
    //     ("particle_count", i(particle_count)),
    //     ("g0", f(g0)),
    //     ("theta", f(theta)),
    //     ("gamma", f(gamma)),
    //     ("alpha", f(alpha)),
    //     ("sigma", f(sigma)),
    //     ("tiled", ParamValue::Bool(false)),
    //     ("elite", ParamValue::Bool(elite)),
    //     ("normalizer", ParamValue::Normalizer(Normalizer::MinMax)),
    //     (
    //       "behavior",
    //       ParamValue::Behavior(Behavior {
    //         edge: edge,
    //         vmax: false,
    //       }),
    //     ),
    //   ],
    //   false,
    // )?;
    // utils::check_cec17::<RgsaParticle, Rgsa<RgsaParticle>>(
    //   "test",
    //   "rgsa_test",
    //   iterations,
    //   dim,
    //   attempts,
    //   vec![
    //     ("particle_count", i(particle_count)),
    //     ("alpha", f(10.)),
    //     ("g0", f(1000.)),
    //     ("tiled", ParamValue::Bool(false)),
    //     ("normalizer", ParamValue::Normalizer(Normalizer::MinMax)),
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
    // utils::check_cec17::<GsaParticle, Gsa<GsaParticle>>(
    //   "gachi_test",
    //   "ogsa_test",
    //   iterations,
    //   dim,
    //   attempts,
    //   vec![
    //     ("particle_count", i(particle_count)),
    //     ("alpha", f(5.)),
    //     ("g0", f(1000.)),
    //     ("tiled", ParamValue::Bool(false)),
    //     ("normalizer", ParamValue::Normalizer(Normalizer::MinMax)),
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

    //utils::check_cec17::<PsoParticle, Pso<PsoParticle>>(
    //  "test",
    //  "pso_test",
    //  iterations,
    //  dim,
    //  attempts,
    //  vec![
    //    ("particle_count", i(particle_count)),
    //    ("w", f(0.5)),
    //    ("phi_p", f(2.)),
    //    ("phi_g", f(2.)),
    //    (
    //      "behavior",
    //      ParamValue::Behavior(Behavior {
    //        edge: Edge::Pass,
    //        vmax: false,
    //      }),
    //    ),
    //  ],
    //  true,
    //)?;
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
    // utils::run_grid_searches::<MgsaParticle, Mgsa<MgsaParticle>>(
    //   "mgsa_test",
    //   attempts,
    //   iterations,
    //   dim,
    //   (
    //     "alpha".to_owned(),
    //     vec![f(1.0), f(2.0), f(5.0), f(10.0), f(20.0), f(50.0), f(100.0)],
    //   ),
    //   (
    //     "g0".to_owned(),
    //     vec![
    //       f(2.0),
    //       f(5.0),
    //       f(10.0),
    //       f(20.0),
    //       f(50.0),
    //       f(100.0),
    //       f(200.0),
    //       f(500.0),
    //       f(1000.0),
    //       f(2000.0),
    //       f(5000.0),
    //       f(10000.0),
    //     ],
    //   ),
    //   vec![
    //     ("particle_count", i(particle_count)),
    //     ("tiled", ParamValue::Tiled(false)),
    //     ("theta", f(1.0)),
    //     ("normalizer", ParamValue::Normalizer(Normalizer::MinMax)),
    //     (
    //       "behavior",
    //       ParamValue::Behavior(Behavior {
    //         edge: Edge::Pass,
    //         vmax: false,
    //       }),
    //     ),
    //   ],
    // )?;
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
