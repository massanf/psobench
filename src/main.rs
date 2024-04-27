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
use optimizers::{gsa::Gsa, pso::Pso};
#[allow(unused_imports)]
use particles::{
  gsa::GsaParticle,
  pso::PsoParticle,
  traits::{Behavior, Edge},
};
use strum::IntoEnumIterator;
#[allow(unused_imports)]
use ParamValue::Float as f;
#[allow(unused_imports)]
use ParamValue::Int as i;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  // let dims = [10, 30, 50, 100];
  let dims = [100];
  let iterations = 1000;
  let attempts = 10;

  // params
  let particle_count = 100;

  // grid
  let grid = false;

  for dim in dims {
    for tiled in [true, false].iter() {
      for normalizer in Normalizer::iter() {
        match grid {
          false => {
            utils::check_cec17::<GsaParticle, Gsa<GsaParticle>>(
              "test",
              generate_name_with_normalizer_and_tiled(normalizer, *tiled).as_str(),
              iterations,
              dim,
              attempts,
              vec![
                ("g0", g0_with_normalizer(normalizer)),
                ("alpha", alpha_with_normalizer(normalizer)),
                ("particle_count", i(particle_count)),
                ("normalizer", ParamValue::Normalizer(normalizer)),
                ("tiled", ParamValue::Tiled(*tiled)),
                ("behavior", generate_behavior_with_tiled(*tiled)),
              ],
              true,
            )?;
          }
          true => {
            utils::run_grid_searches::<GsaParticle, Gsa<GsaParticle>>(
              generate_name_with_normalizer_and_tiled(normalizer, *tiled).as_str(),
              attempts,
              iterations,
              dim,
              parameters::GSA_G0_OPTIONS.clone(),
              parameters::GSA_ALPHA_OPTIONS.clone(),
              vec![
                ("particle_count", i(particle_count)),
                ("normalizer", ParamValue::Normalizer(normalizer)),
                ("tiled", ParamValue::Tiled(*tiled)),
                ("behavior", generate_behavior_with_tiled(*tiled)),
              ],
            )?;
          }
        }
      }
    }
  }

  Ok(())
}

fn generate_behavior_with_tiled(tiled: bool) -> ParamValue {
  ParamValue::Behavior(Behavior {
    edge: match tiled {
      true => Edge::Cycle,
      false => Edge::Pass,
    },
    vmax: false,
  })
}

fn g0_with_normalizer(normalizer: Normalizer) -> ParamValue {
  ParamValue::Float(match normalizer {
    Normalizer::MinMax => 1000.,
    _ => 50.,
  })
}

fn alpha_with_normalizer(_normalizer: Normalizer) -> ParamValue {
  ParamValue::Float(5.)
}

fn generate_name_with_normalizer_and_tiled(normalizer: Normalizer, tiled: bool) -> String {
  match tiled {
    true => format!("gsa_{:?}_tiled", normalizer),
    false => format!("gsa_{:?}", normalizer),
  }
}
