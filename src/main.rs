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
  // let dims = [10, 100];
  let dims = [50];
  let iterations = 1000;
  let attempts = 10;

  // params
  let particle_count = 50;

  // grid
  let grid = true;

  for dim in dims {
    for normalizer in Normalizer::iter() {
      for tiled in [false, true].iter() {
        if normalizer != Normalizer::ZScore {
          continue;
        }
        match grid {
          false => {
            utils::check_cec17::<GsaParticle, Gsa<GsaParticle>>(
              "test",
              utils::name_from_normalizer_and_tiled(normalizer, *tiled).as_str(),
              iterations,
              dim,
              attempts,
              vec![
                ("g0", utils::g0_from_normalizer(normalizer)),
                ("alpha", utils::alpha_from_normalizer(normalizer)),
                ("particle_count", i(particle_count)),
                ("normalizer", ParamValue::Normalizer(normalizer)),
                ("tiled", ParamValue::Tiled(*tiled)),
                ("behavior", utils::behavior_from_tiled(*tiled)),
              ],
              true,
            )?;
          }
          true => {
            utils::run_grid_searches::<GsaParticle, Gsa<GsaParticle>>(
              utils::name_from_normalizer_and_tiled(normalizer, *tiled).as_str(),
              attempts,
              iterations,
              dim,
              parameters::GSA_G0_OPTIONS.clone(),
              parameters::GSA_ALPHA_OPTIONS.clone(),
              vec![
                ("particle_count", i(particle_count)),
                ("normalizer", ParamValue::Normalizer(normalizer)),
                ("tiled", ParamValue::Tiled(*tiled)),
                ("behavior", utils::behavior_from_tiled(*tiled)),
              ],
            )?;
          }
        }
      }
    }
  }

  Ok(())
}
