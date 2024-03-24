extern crate nalgebra as na;
extern crate rand;
mod optimization_problem;
use std::path::PathBuf;
mod functions;
mod particle_trait;
use crate::pso_trait::Param;
// use crate::pso_trait::PSOTrait;
mod pso;
mod pso_trait;
mod utils;
use pso::particle::Particle;
use pso::pso::PSO;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  //   // Problem Settings
  //   let mut problem_set = Vec::new();
  //   for i in 1..=5 {
  //     if i == 2 {
  //       continue;
  //     }
  //     problem_set.push(functions::cec17(i, 30));
  //   }
  let problem = functions::f1(30);

  // Experiment Settings
  //   let particle_count = 30;
  let particle_count = 30;
  let iterations = 1000;

  let out_directory = PathBuf::from("data/base_pso_test4");

  utils::grid_search::<'_, Particle, PSO<'_, Particle>>(
    iterations,
    &problem,
    2,
    (
      "phi_p".to_owned(),
      utils::Range {
        min: Param::Numeric(-4.0),
        max: Param::Numeric(4.0),
        step: Param::Numeric(0.4),
      },
    ),
    (
      "phi_g".to_owned(),
      utils::Range {
        min: Param::Numeric(-4.0),
        max: Param::Numeric(4.0),
        step: Param::Numeric(0.4),
      },
    ),
    [
      ("w".to_owned(), Param::Numeric(0.8)),
      ("particle_count".to_owned(), Param::Count(particle_count)),
    ]
    .iter()
    .cloned()
    .collect(),
    out_directory,
  )?;

  //   let params = vec![
  //     ("particle_count".to_owned(), Param::Count(particle_count)),
  //     ("phi_p".to_owned(), Param::Numeric(1.0)),
  //     ("phi_g".to_owned(), Param::Numeric(1.0)),
  //     ("w".to_owned(), Param::Numeric(0.8)),
  //   ]
  //   .iter()
  //   .cloned()
  //   .collect();

  //   let mut pso: PSO<'_, Particle> = PSO::new(
  //     "PSO",
  //     &problem,
  //     params,
  //     out_directory.join(format!(
  //       "g_1_d_2" //   "{}/{}={:.2},{}={:.2}/{}",
  //                 //   problem.name(),
  //                 //   search_params[0].0.clone(),
  //                 //   p,
  //                 //   search_params[1].0.clone(),
  //                 //   g,
  //                 //   attempt
  //     )),
  //   );
  //   pso.run(iterations);
  //   pso.save_data()?;
  //   pso.save_summary()?;
  //   pso.save_config()?;
  Ok(())
}
