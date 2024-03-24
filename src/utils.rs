use crate::optimization_problem;
use crate::particle_trait::ParticleTrait;
use crate::pso_trait::PSOTrait;
use crate::pso_trait::Param;
use indicatif::ProgressBar;
use nalgebra::DVector;
use optimization_problem::Problem;
use rand::distributions::{Distribution, Uniform};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

#[allow(dead_code)]
pub struct Range {
  pub min: Param,
  pub max: Param,
  pub step: Param,
}

pub fn uniform_distribution(low: &DVector<f64>, high: &DVector<f64>) -> DVector<f64> {
  let mut rng = rand::thread_rng();
  DVector::from_iterator(
    low.len(),
    (0..low.len()).map(|i| Uniform::new(low[i], high[i]).sample(&mut rng)),
  )
}

// TODO: There must be a better place to put this.
pub fn random_init_pos(problem: &Problem) -> DVector<f64> {
  let b_lo: DVector<f64> = DVector::from_element(problem.dim(), problem.domain().0);
  let b_up: DVector<f64> = DVector::from_element(problem.dim(), problem.domain().1);
  uniform_distribution(&b_lo, &b_up)
}

// TODO: There must be a better place to put this.
pub fn random_init_vel(problem: &Problem) -> DVector<f64> {
  let b_lo: DVector<f64> = DVector::from_element(problem.dim(), problem.domain().0);
  let b_up: DVector<f64> = DVector::from_element(problem.dim(), problem.domain().1);

  uniform_distribution(
    &DVector::from_iterator(problem.dim(), (&b_up - &b_lo).iter().map(|b| -b.abs())),
    &DVector::from_iterator(problem.dim(), (&b_up - &b_lo).iter().map(|b| b.abs())),
  )
}

pub fn create_directory(path: PathBuf) {
  // Handle output directory creation / deletion
  if path.exists() {
    println!("The directory {:?} already exists. Overwrite? (y/n)", path);
    let mut user_input = String::new();
    let _ = io::stdin().read_line(&mut user_input);

    match user_input.trim().to_lowercase().as_str() {
      "y" => {
        let _ = fs::remove_dir_all(path.clone());
      }
      _ => {
        println!("Cancelled.");
        std::process::exit(1);
      }
    }
  }
  let _ = fs::create_dir_all(path);
}

#[allow(dead_code)]
pub fn grid_search<'a, U: ParticleTrait, T: PSOTrait<U>>(
  iterations: usize,
  problem: &'a Problem,
  attempts: usize,
  param1: (String, Range),
  param2: (String, Range),
  base_params: HashMap<String, Param>,
  out_directory: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
  let cnt1: usize;
  let cnt2: usize;
  match (&param1.1.min, &param1.1.max, &param1.1.step) {
    (Param::Numeric(min), Param::Numeric(max), Param::Numeric(step)) => {
      cnt1 = ((max - min) / step) as usize;
    }
    (Param::Count(min), Param::Count(max), Param::Count(step)) => {
      cnt1 = ((max - min) / step) as usize;
    }
    _ => {
      eprintln!("Range variable types do not match.");
      std::process::exit(1);
    }
  }
  match (&param2.1.min, &param2.1.max, &param2.1.step) {
    (Param::Numeric(min), Param::Numeric(max), Param::Numeric(step)) => {
      cnt2 = ((max - min) / step) as usize;
    }
    (Param::Count(min), Param::Count(max), Param::Count(step)) => {
      cnt2 = ((max - min) / step) as usize;
    }
    _ => {
      eprintln!("Range variable types do not match.");
      std::process::exit(1);
    }
  }

  create_directory(out_directory.clone());
  let bar = ProgressBar::new(((cnt1 + 1) * (cnt2 + 1) * attempts) as u64);
  for idx1 in 0..=cnt1 {
    for idx2 in 0..=cnt2 {
      let x1: Param;
      let x2: Param;
      match (&param1.1.min, &param1.1.step) {
        (Param::Numeric(min), Param::Numeric(step)) => {
          x1 = Param::Numeric(min + step * idx1 as f64);
        }
        (Param::Count(min), Param::Count(step)) => {
          x1 = Param::Count(min + step * idx1 as isize);
        }
        _ => {
          eprintln!("Range variable types do not match.");
          std::process::exit(1);
        }
      }
      match (&param2.1.min, &param2.1.step) {
        (Param::Numeric(min), Param::Numeric(step)) => {
          x2 = Param::Numeric(min + step * idx2 as f64);
        }
        (Param::Count(min), Param::Count(step)) => {
          x2 = Param::Count(min + step * idx2 as isize);
        }
        _ => {
          eprintln!("Range variable types do not match.");
          std::process::exit(1);
        }
      }
      for attempt in 0..attempts {
        let mut params = base_params.clone();
        params.insert(param1.0.clone(), x1.clone());
        params.insert(param2.0.clone(), x2.clone());
        let mut pso: T = T::new(
          "PSO",
          problem.clone(),
          params,
          out_directory.join(format!(
            "{}/{}={},{}={}/{}",
            problem.name(),
            param1.0.clone(),
            x1,
            param2.0.clone(),
            x2,
            attempt
          )),
        );
        pso.run(iterations);
        pso.save_summary()?;
        pso.save_config()?;
        bar.inc(1);
      }
    }
  }
  Ok(())
}

#[allow(dead_code)]
pub fn grid_search_dim<U: ParticleTrait, T: PSOTrait<U>>(
  iterations: usize,
  problem_type: Arc<dyn Fn(usize) -> Problem>,
  attempts: usize,
  dim: Range,
  param: (String, Range),
  base_params: HashMap<String, Param>,
  out_directory: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
  let cnt1: usize;
  let cnt_dim: usize;

  match (&param.1.min, &param.1.max, &param.1.step) {
    (Param::Numeric(min), Param::Numeric(max), Param::Numeric(step)) => {
      cnt1 = ((max - min) / step) as usize;
    }
    (Param::Count(min), Param::Count(max), Param::Count(step)) => {
      cnt1 = ((max - min) / step) as usize;
    }
    _ => {
      eprintln!("Range variable types do not match.");
      std::process::exit(1);
    }
  }

  match (&dim.min, &dim.max, &dim.step) {
    (Param::Count(min), Param::Count(max), Param::Count(step)) => {
      cnt_dim = ((max - min) / step) as usize;
    }
    _ => {
      eprintln!("Range variable types do not match.");
      std::process::exit(1);
    }
  }

  create_directory(out_directory.clone());
  let bar = ProgressBar::new(((cnt1 + 1) * (cnt_dim + 1) * attempts) as u64);
  for idx1 in 0..=cnt1 {
    for idx_dim in 0..=cnt_dim {
      let x1: Param;
      let x_dim: usize;
      match (&param.1.min, &param.1.step) {
        (Param::Numeric(min), Param::Numeric(step)) => {
          x1 = Param::Numeric(min + step * idx1 as f64);
        }
        (Param::Count(min), Param::Count(step)) => {
          x1 = Param::Count(min + step * idx1 as isize);
        }
        _ => {
          eprintln!("Range variable types do not match.");
          std::process::exit(1);
        }
      }
      match (&dim.min, &dim.step) {
        (Param::Count(min), Param::Count(step)) => {
          x_dim = (min + step * idx_dim as isize) as usize;
        }
        _ => {
          eprintln!("Range variable types do not match.");
          std::process::exit(1);
        }
      }
      let problem = problem_type(x_dim);
      for attempt in 0..attempts {
        let mut params = base_params.clone();
        params.insert(param.0.clone(), x1.clone());
        let mut pso: T = T::new(
          "PSO",
          problem.clone(),
          params,
          out_directory.join(format!(
            "{}/{}={},{}={}/{}",
            problem_type(x_dim).name(),
            param.0.clone(),
            x1,
            "dim",
            x_dim,
            attempt
          )),
        );
        pso.run(iterations);
        pso.save_summary()?;
        pso.save_config()?;
        bar.inc(1);
      }
    }
  }
  Ok(())
}
