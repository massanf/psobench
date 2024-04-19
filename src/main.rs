extern crate nalgebra as na;
extern crate rand;
mod functions;
mod grid_search;
mod optimizers;
mod parameters;
mod particles;
mod problems;
mod utils;
use crate::optimizers::traits::ParamValue;
#[allow(unused_imports)]
use optimizers::{gsa::Gsa, pso::Pso, tiled_gsa::TiledGsa};
#[allow(unused_imports)]
use particles::{
  gsa_particle::GsaParticle,
  igsa_particle::IgsaParticle,
  pso_particle::PsoParticle,
  traits::{Behavior, Edge},
};
#[allow(unused_imports)]
use ParamValue::Float as f;
#[allow(unused_imports)]
use ParamValue::Int as i;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let dims = [10, 30, 50, 100];
  // let dim = 30;
  let iterations = 1000;
  let attempts = 10;

  for dim in dims {
    println!("{}", dim);
    let scale = 10;
    let pc: isize = dim as isize * scale;

    // gsa
    utils::check_cec17::<GsaParticle, Gsa<GsaParticle>>(
      format!("test_{}", iterations).as_str(),
      "gsa",
      iterations,
      dim,
      attempts,
      Behavior {
        edge: Edge::Pass,
        vmax: false,
      },
      vec![("g0", f(500.0)), ("alpha", f(5.0)), ("particle_count", i(pc))],
    )?;

    // tiled gsa
    utils::check_cec17::<GsaParticle, TiledGsa<GsaParticle>>(
      format!("test_{}", iterations).as_str(),
      "tiledgsa",
      iterations,
      dim,
      attempts,
      Behavior {
        edge: Edge::Cycle,
        vmax: false,
      },
      vec![("g0", f(1000.0)), ("alpha", f(5.0)), ("particle_count", i(pc))],
    )?;

    //   // igsa
    //   utils::check_cec17::<IgsaParticle, Gsa<IgsaParticle>>(
    //     format!("test_{}", iterations).as_str(),
    //     "igsa",
    //     iterations,
    //     dim,
    //     attempts,
    //     Behavior {
    //       edge: Edge::Pass,
    //       vmax: false,
    //     },
    //     vec![("g0", f(500.0)), ("alpha", f(5.0)), ("particle_count", i(pc))],
    //   )?;

    //   // tiled igsa
    //   utils::check_cec17::<IgsaParticle, TiledGsa<IgsaParticle>>(
    //     format!("test_{}", iterations).as_str(),
    //     "tiledigsa",
    //     iterations,
    //     dim,
    //     attempts,
    //     Behavior {
    //       edge: Edge::Cycle,
    //       vmax: false,
    //     },
    //     vec![("g0", f(1000.0)), ("alpha", f(5.0)), ("particle_count", i(pc))],
    //   )?;
  }

  // utils::check_cec17::<GsaParticle, Gsa<GsaParticle>>(
  //   format!("test_{}", iterations).as_str(),
  //   "gsa",
  //   iterations,
  //   dim,
  //   attempts,
  //   behavior,
  //   vec![("g0", f(100.0)), ("alpha", f(20.0)), ("particle_count", i(30))],
  // )?;

  // utils::check_cec17::<GsaParticle, TiledGsa<GsaParticle>>(
  //   format!("test_{}", iterations).as_str(),
  //   "tiledgsa",
  //   iterations,
  //   dim,
  //   attempts,
  //   behavior,
  //   vec![("g0", f(1000.0)), ("alpha", f(5.0)), ("particle_count", i(30))],
  // )?;

  // utils::check_cec17::<PsoParticle, Pso<PsoParticle>>(
  //   "test",
  //   "pso",
  //   iterations,
  //   dim,
  //   attempts,
  //   behavior,
  //   vec![
  //     ("w", f(0.8)),
  //     ("phi_p", f(1.)),
  //     ("phi_g", f(1.)),
  //     ("particle_count", i(30)),
  //   ],
  // )?;

  // utils::run_grid_searches::<GsaParticle, Gsa<GsaParticle>>(
  //   "gsa",
  //   attempts,
  //   iterations,
  //   dim,
  //   parameters::GSA_G0_OPTIONS.clone(),
  //   parameters::GSA_ALPHA_OPTIONS.clone(),
  //   vec![("particle_count", i(30))],
  //   behavior,
  // )?;

  Ok(())
}
