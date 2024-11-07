use crate::optimizers::gsa::Normalizer;
use crate::optimizers::traits::{
  Data, DataExporter, GlobalBestPos, Name, OptimizationProblem, Optimizer, ParamValue, Particles,
};
use crate::particles::traits::{Behavior, Mass, Particle, Position, Velocity};
use crate::problems;
// use crate::rand::Rng;
use crate::rand::Rng;
use crate::utils;
use nalgebra::DVector;
use problems::Problem;
use rayon::prelude::*;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::mem;
use std::path::PathBuf;

#[derive(Clone)]
pub struct Mgsa<T> {
  name: String,
  problem: Problem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  global_worst_pos: Option<DVector<f64>>,
  g: f64,
  data: Vec<(f64, f64, Option<Vec<T>>)>,
  out_directory: PathBuf,
  g0: f64,
  alpha: f64,
  theta: f64,
  gamma: f64,
  sigma: f64,
  elite: bool,
  save: bool,
  normalizer: Normalizer,
}

impl<T: Particle + Position + Velocity + Mass + Clone> Optimizer<T> for Mgsa<T> {
  fn new(
    name: String,
    problem: Problem,
    parameters: HashMap<String, ParamValue>,
    out_directory: PathBuf,
    save: bool,
  ) -> Mgsa<T> {
    assert!(
      parameters.contains_key("particle_count"),
      "Key 'particle_count' not found."
    );
    let number_of_particles = match parameters["particle_count"] {
      ParamValue::Int(val) => val as usize,
      _ => {
        eprintln!("Error: parameter 'particle_count' should be of type Param::Int.");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("g0"), "Key 'g0' not found.");
    let g0 = match parameters["g0"] {
      ParamValue::Float(val) => val,
      _ => {
        eprintln!("Error: parameter 'g0' should be of type Param::Float.");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("alpha"), "Key 'alpha' not found.");
    let alpha = match parameters["alpha"] {
      ParamValue::Float(val) => val,
      _ => {
        eprintln!("Error: parameter 'alpha' should be of type Param::Float.");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("normalizer"), "Key 'normalizer' not found.");
    let normalizer = match parameters["normalizer"] {
      ParamValue::Normalizer(val) => val,
      _ => {
        eprintln!("Error: parameter 'normalizer' should be of type Param::Normalizer.");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("behavior"), "Key 'behavior' not found.");
    let behavior = match parameters["behavior"] {
      ParamValue::Behavior(val) => val,
      _ => {
        eprintln!("Error: parameter 'behavior' should be of type Param::Behavior.");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("theta"), "Key 'theta' not found.");
    let theta = match parameters["theta"] {
      ParamValue::Float(val) => val,
      _ => {
        eprintln!("Error: parameter 'theta' should be of type Param::float.");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("gamma"), "Key 'gamma' not found.");
    let gamma = match parameters["gamma"] {
      ParamValue::Float(val) => val,
      _ => {
        eprintln!("Error: parameter 'gamma' should be of type Param::float.");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("elite"), "Key 'elite' not found.");
    let elite = match parameters["elite"] {
      ParamValue::Bool(val) => val,
      _ => {
        eprintln!("Error: parameter 'elite' should be of type Param::bool.");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("sigma"), "Key 'sigma' not found.");
    let sigma = match parameters["sigma"] {
      ParamValue::Float(val) => val,
      _ => {
        eprintln!("Error: parameter 'sigma' should be of type Param::f64.");
        std::process::exit(1);
      }
    };

    let mut mgsa = Mgsa {
      name,
      problem,
      particles: Vec::new(),
      global_best_pos: None,
      global_worst_pos: None,
      g: g0,
      data: Vec::new(),
      out_directory,
      g0,
      alpha,
      theta,
      gamma,
      sigma,
      elite,
      save,
      normalizer,
    };

    mgsa.init(number_of_particles, behavior);
    mgsa
  }

  fn init(&mut self, number_of_particles: usize, behavior: Behavior) {
    let problem = &mut self.problem();
    let mut particles: Vec<T> = Vec::new();
    for _ in 0..number_of_particles {
      particles.push(T::new(problem, behavior));
    }

    let mut global_best_pos = None;
    let mut global_worst_pos = None;
    for particle in particles.clone() {
      if global_best_pos.is_none() || problem.f(particle.pos()) < problem.f(global_best_pos.as_ref().unwrap()) {
        global_best_pos = Some(particle.pos().clone());
      }
      if global_worst_pos.is_none() || problem.f(particle.pos()) > problem.f(global_worst_pos.as_ref().unwrap()) {
        global_worst_pos = Some(particle.pos().clone());
      }
    }

    self.particles = particles;
    self.set_global_best_pos(global_best_pos.unwrap());
    self.set_global_worst_pos(global_worst_pos.unwrap());

    utils::create_directory(self.out_directory().to_path_buf(), true, false);
  }

  fn calculate_vel(&mut self, _i: usize) -> DVector<f64> {
    panic!("deprecated");
  }

  fn run(&mut self, iterations: usize) {
    let n = self.particles().len();

    let mut initial_spread = None;
    let mut x_record: Vec<Vec<DVector<f64>>> = Vec::new();
    let mut f_record: Vec<Vec<f64>> = Vec::new();

    for iter in 0..iterations {
      println!("--{}--", iter);
      x_record = Vec::new();
      f_record = Vec::new();
      let mut distances = Vec::new();
      for i in 0..n {
        for j in 0..n {
          if i == j {
            continue;
          }
          let distance = (self.particles()[i].pos() - self.particles()[j].pos()).norm();
          distances.push(distance);
        }
      }
      let use_avg = false;
      let spread = match use_avg {
        true => distances.iter().sum::<f64>() / distances.len() as f64,
        false => utils::calculate_std(&distances),
      };
      if initial_spread.is_none() {
        initial_spread = Some(spread);
      }
      println!("spread: {}", spread / initial_spread.unwrap());
      // let ratio = spread / initial_spread.unwrap();

      let ratio = (-self.alpha * iter as f64 / iterations as f64).exp();
      // let iteration_ratio = 1. - iter as f64 / iterations as f64;

      self.g = self.g0 * ratio;

      let mut fitness = Vec::new();
      for idx in 0..n {
        let pos = self.particles()[idx].pos().clone();
        fitness.push(self.problem().f(&pos));
      }

      f_record.push(fitness);
      let m_record = utils::original_gsa_mass_with_record(f_record.clone(), 100);

      // let m = match self.normalizer {
      //   Normalizer::MinMax => utils::original_gsa_mass(fitness),
      //   Normalizer::ZScore => utils::z_mass(fitness),
      //   Normalizer::Robust => utils::robust_mass(fitness),
      //   Normalizer::Rank => utils::rank_mass(fitness),
      //   Normalizer::Sigmoid2 => utils::sigmoid2_mass(fitness),
      //   Normalizer::Sigmoid4 => utils::sigmoid4_mass(fitness),
      // };
      // let m: Vec<f64> = fitness.iter().map(|&f| 1. / f).collect();

      for (mass, particle) in m_record[m_record.len() - 1].iter().zip(self.particles_mut().iter_mut()) {
        particle.set_mass(*mass);
      }

      // Calculate vels.
      let mut x: Vec<DVector<f64>> = Vec::new();
      let mut v = Vec::new();
      for idx in 0..n {
        x.push(self.particles()[idx].pos().clone());
        v.push(self.particles()[idx].vel().clone());
      }
      x_record.push(x);

      let vels = calculate_vels(
        x_record.clone(),
        m_record.clone(),
        &v,
        self.g,
        iter as f64 / iterations as f64,
        ratio,
        self.theta,
        self.gamma,
        self.elite,
        self.sigma,
      );

      // Clear memory.
      self.problem().clear_memo();

      // Update the position, best and worst.
      let mut new_global_best_pos = None;
      let mut new_global_worst_pos = None;
      // for (i, m_i) in 0.. {
      for (i, vel) in vels.iter().enumerate().take(n) {
        let mut temp_problem = mem::take(&mut self.problem);
        let particle = &mut self.particles_mut()[i];
        particle.update_vel(vel.clone(), &mut temp_problem);
        particle.move_pos(&mut temp_problem);
        let pos = particle.pos().clone();

        // Update best.
        if new_global_best_pos.is_none()
          || (temp_problem.f(&pos) < temp_problem.f(&new_global_best_pos.clone().unwrap()))
        {
          new_global_best_pos = Some(pos.clone());
        }

        // Update worst.
        if new_global_worst_pos.is_none()
          || (temp_problem.f(&pos) > temp_problem.f(&new_global_worst_pos.clone().unwrap()))
        {
          new_global_worst_pos = Some(pos.clone());
        }
        self.problem = temp_problem;
      }
      self.update_global_best_pos(new_global_best_pos.clone().unwrap());
      self.update_global_worst_pos(new_global_worst_pos.unwrap());

      // Save the data for current iteration.
      let gbest = self.problem.f(&self.global_best_pos());
      let gworst = self.problem.f(&self.global_worst_pos());

      let particles = self.particles.clone();
      self.add_data(self.save, gbest, gworst, particles);
    }
  }
}

fn calculate_vels(
  x: Vec<Vec<DVector<f64>>>,
  f: Vec<Vec<f64>>,
  v: &Vec<DVector<f64>>,
  large_g: f64,
  progress: f64,
  spread: f64,
  theta: f64,
  gamma: f64,
  elite: bool,
  sigma: f64,
) -> Vec<DVector<f64>> {
  let t = x.len();
  let n = x[0].len();
  let d = x[0][0].len();

  let vels: Vec<DVector<f64>> = (0..n)
    // .into_par_iter()
    .into_iter()
    .map(|k| {
      // for k in 0..n {
      let mut a: DVector<f64> = DVector::from_element(d, 0.);

      for l in 0..t {
        let p = std::cmp::min(std::cmp::max((n as f64 * (1. - progress as f64)) as usize, 1), n);
        let mut sorted_f = f[l].clone();
        sorted_f.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let p_largest: Vec<f64> = sorted_f.iter().take(p).copied().collect();
        let influences: Vec<bool> = match elite {
          true => f[l].iter().map(|x| p_largest.contains(x)).collect(),
          false => vec![true; n],
        };

        let mut sum_fg = 0.;
        let mut sum_g = 0.;
        for j in 0..n {
          if !influences[j] {
            continue;
          }
          let g_jk = mock_gaussian(&x[l][j], &x[t - 1][k], spread, sigma);
          sum_fg += f[l][j] * g_jk;
          sum_g += g_jk;
        }

        for i in 0..n {
          if (l == t - 1 && i == k) || !influences[i] {
            continue;
          }
          let r: DVector<f64> = &x[l][i] - &x[t - 1][k];

          let gravity = f[l][i] / (r.norm() + std::f64::EPSILON) * r;
          // let repellent = gamma * (1. - progress * theta) * g / sum_g;

          let mut a_delta = gravity;
          // let a_delta = gravity - repellent;
          
          let g = mock_gaussian(&x[l][i], &x[t - 1][k], spread, sigma);

          let gravity = g / sum_fg;
          let repellent = gamma * (1. - progress * theta) * g / sum_g;
          println!("---");
          println!("f: {}", f[l][i]);
          println!("spread: {}", spread);
          println!("sigma: {}", sigma);
          println!("gravity: {}", gravity);
          println!("repellent: {}", repellent);

          let mut rng = rand::thread_rng();
          for e in a_delta.iter_mut() {
            let rand: f64 = rng.gen_range(0.0..1.0);
            *e *= rand;
          }

          a += large_g * a_delta;
        }
      }
      let mut rng = rand::thread_rng();
      let rand: f64 = rng.gen_range(0.0..1.0);
      rand * v[k].clone() + a
    })
    .collect();
  println!(
    "vel: {}",
    vels.iter().map(|x| x.norm()).sum::<f64>() / vels.len() as f64
  );
  vels
}

// Using Gaussian.
// fn calculate_vels(
//   x: Vec<Vec<DVector<f64>>>,
//   f: Vec<Vec<f64>>,
//   _v: &Vec<DVector<f64>>,
//   large_g: f64,
//   progress: f64,
//   spread: f64,
//   theta: f64,
//   gamma: f64,
//   elite: bool,
//   sigma: f64,
// ) -> Vec<DVector<f64>> {
//   let t = x.len();
//   let n = x[0].len();
//   let d = x[0][0].len();
//
//   let vels: Vec<DVector<f64>> = (0..n)
//     .into_par_iter()
//     .map(|k| {
//       // for k in 0..n {
//       let mut a: DVector<f64> = DVector::from_element(d, 0.);
//
//       for l in 0..t {
//         let p = std::cmp::min(std::cmp::max((n as f64 * (1. - progress as f64)) as usize, 1), n);
//         let mut sorted_f = f[l].clone();
//         sorted_f.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
//         let p_largest: Vec<f64> = sorted_f.iter().take(p).copied().collect();
//         let influences: Vec<bool> = match elite {
//           true => f[l].iter().map(|x| p_largest.contains(x)).collect(),
//           false => vec![true; n],
//         };
//
//         let mut sum_fg = 0.;
//         let mut sum_g = 0.;
//         for j in 0..n {
//           if !influences[j] {
//             continue;
//           }
//           let g_jk = mock_gaussian(&x[l][j], &x[t - 1][k], spread, sigma);
//           sum_fg += f[l][j] * g_jk;
//           sum_g += g_jk;
//         }
//
//         if sum_g == 0. || sum_fg == 0. {
//           // This happens when particle k is a loner.
//           // I think we can safely ignore.
//           println!("warn: 0 sum.");
//           continue;
//         }
//
//         for i in 0..n {
//           if !influences[i] {
//             continue;
//           }
//           let r: DVector<f64> = &x[l][i] - &x[t - 1][k];
//           let g = mock_gaussian(&x[l][i], &x[t - 1][k], spread, sigma);
//
//           let gravity = f[l][i] * g / sum_fg;
//           let repellent = gamma * (1. - progress * theta) * g / sum_g;
//
//           let a_delta = (gravity - repellent) * r;
//
//           a += large_g * a_delta;
//         }
//       }
//       a
//     })
//     .collect();
//   println!(
//     "vel: {}",
//     vels.iter().map(|x| x.norm()).sum::<f64>() / vels.len() as f64
//   );
//   vels
// }


fn mock_gaussian(i: &DVector<f64>, j: &DVector<f64>, spread: f64, sigma: f64) -> f64 {
  if !spread.is_finite() {
    return 1.;
  }
  let d = i.len();
  let variance = (sigma * spread).powi(2);
  let constant = 1.0 / ((2.0 * std::f64::consts::PI * variance).powf(d as f64 / 2.));
  let res = constant * (-(i - j).norm_squared() / (2. * variance)).exp();
  if !res.is_finite() {
    panic!("infinite gaussian: {}", spread);
  }
  res
}

// fn _calculate_vels(
//   x: Vec<DVector<f64>>,
//   _v: Vec<DVector<f64>>,
//   f: Vec<f64>,
//   g: f64,
//   progress: f64,
//   avg_dist_rate: f64,
//   theta: f64,
//   gamma: f64,
//   _elite: bool,
//   sigma: f64,
// ) -> Vec<DVector<f64>> {
//   let n = x.len();
//   let d = x[0].len();
//   let mut vels = Vec::with_capacity(n);
//
//   // let k = std::cmp::min(std::cmp::max((n as f64 * (1. - progress as f64)) as usize, 1), n);
//   // let mut sorted_f = f.clone();
//   // sorted_f.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
//   // let k_largest: Vec<f64> = sorted_f.iter().take(k).copied().collect();
//   // let influences: Vec<bool> = match elite {
//   //   true => f.iter().map(|x| k_largest.contains(x)).collect(),
//   //   false => vec![true; n],
//   // };
//
//   // let mut vec = Vec::new();
//   for k in 0..n {
//     //assert!(i < n);
//     let mut a: DVector<f64> = DVector::from_element(d, 0.);
//     // let mut rng = rand::thread_rng();
//
//     let mut sum_fg = 0.;
//     let mut sum_g = 0.;
//     for j in 0..n {
//       // if !influences[j] {
//       //   continue;
//       // }
//       sum_fg += f[j] * mock_gaussian(&x[j], &x[k], avg_dist_rate, sigma);
//       sum_g += 1. * mock_gaussian(&x[j], &x[k], avg_dist_rate, sigma);
//     }
//
//     for i in 0..n {
//       // if k == i || !influences[i] {
//       if k == i {
//         continue;
//       }
//
//       let r = x[i].clone() - x[k].clone();
//       let dist = mock_gaussian(&x[i], &x[k], avg_dist_rate, sigma);
//
//       // let mut gravity = match influences[j] {
//       //   true => f[j] / ((r.norm() + 1.) * sum_fg),
//       //   false => 0.,
//       // };
//       let gravity = f[i] * dist / sum_fg;
//       // for e in gravity.iter_mut() {
//       //   let rand: f64 = rng.gen_range(0.0..2.0);
//       //   *e *= rand;
//       // }
//
//       // let mut repellent = gamma * (1. - progress * theta) / ((r.norm() + 1.) * sum_g);
//       let repellent = gamma * (1. - progress * theta) * 1. * dist / sum_g;
//       // for e in repellent.iter_mut() {
//       //   let rand: f64 = rng.gen_range(0.0..2.0);
//       //   *e *= rand;
//       // }
//
//       let a_delta = (gravity - repellent) * r.clone();
//       // a_delta *= rng.gen_range(-1.0..3.0);
//       // for e in a_delta.iter_mut() {
//       //   let rand: f64 = rng.gen_range(0.0..2.0);
//       //   *e *= rand;
//       // }
//
//       a += g * a_delta;
//     }
//
//     // let rand: f64 = rng.gen_range(0.0..1.0);
//     // vels.push(rand * v[i].clone() + a);
//     // vec.push(a.clone());
//     vels.push(a);
//   }
//   // println!("{}", vels.iter().map(|x| x.norm()).sum::<f64>() / vels.len() as f64);
//   // let avg = vels.iter().sum::<DVector<f64>>() / vels.len() as f64;
//
//   // let mut no_movement_vels = vels.clone();
//   // for i in 0..vels.len() {
//   //   no_movement_vels[i] = vels[i].clone() - avg.clone();
//   // }
//   vels
//   // no_movement_vels
// }

impl<T> Particles<T> for Mgsa<T> {
  fn particles(&self) -> &Vec<T> {
    &self.particles
  }

  fn particles_mut(&mut self) -> &mut Vec<T> {
    &mut self.particles
  }
}

impl<T> GlobalBestPos for Mgsa<T> {
  fn global_best_pos(&self) -> DVector<f64> {
    self.global_best_pos.clone().unwrap()
  }

  fn global_worst_pos(&self) -> DVector<f64> {
    self.global_worst_pos.clone().unwrap()
  }

  fn option_global_best_pos(&self) -> &Option<DVector<f64>> {
    &self.global_best_pos
  }

  fn option_global_worst_pos(&self) -> &Option<DVector<f64>> {
    &self.global_worst_pos
  }

  fn set_global_best_pos(&mut self, pos: DVector<f64>) {
    self.global_best_pos = Some(pos);
  }

  fn set_global_worst_pos(&mut self, pos: DVector<f64>) {
    self.global_worst_pos = Some(pos);
  }
}

impl<T> OptimizationProblem for Mgsa<T> {
  fn problem(&mut self) -> &mut Problem {
    &mut self.problem
  }
}

impl<T> Name for Mgsa<T> {
  fn name(&self) -> &String {
    &self.name
  }
}

impl<T: Clone> Data<T> for Mgsa<T> {
  fn data(&self) -> &Vec<(f64, f64, Option<Vec<T>>)> {
    &self.data
  }

  fn add_data_impl(&mut self, datum: (f64, f64, Option<Vec<T>>)) {
    self.data.push(datum);
  }
}

impl<T: Position + Velocity + Mass + Clone> DataExporter<T> for Mgsa<T> {
  fn out_directory(&self) -> &PathBuf {
    &self.out_directory
  }

  fn save_data(&mut self) -> Result<(), Box<dyn std::error::Error>> {
    // Serialize it to a JSON string
    let mut vec_data = Vec::new();
    for t in 0..self.data().len() {
      let mut iter_data = Vec::new();
      let datum = self.data()[t].2.clone().unwrap();
      for particle_datum in &datum {
        let pos = particle_datum.pos().clone();
        iter_data.push(json!({
          "fitness": self.problem().f_no_memo(&pos),
          "vel": particle_datum.vel().as_slice(),
          "pos": particle_datum.pos().as_slice(),
          "mass": particle_datum.mass(),
        }));
      }
      vec_data.push(json!({
        "global_best_fitness": self.data()[t].0,
        "global_worst_fitness": self.data()[t].1,
        "particles": iter_data
      }));
    }

    let serialized = serde_json::to_string(&json!(vec_data))?;

    fs::write(self.out_directory().join("data.json"), serialized)?;
    Ok(())
  }
}

#[allow(dead_code)]
fn calculate_mean(data: Vec<f64>) -> f64 {
  data.iter().sum::<f64>() / (data.len() as f64)
}

#[allow(dead_code)]
fn calculate_variance(data: Vec<f64>, mean: f64) -> f64 {
  data
    .iter()
    .map(|value| {
      let diff = value - mean;
      diff * diff
    })
    .sum::<f64>()
    / ((data.len() - 1) as f64) // Sample variance
}

#[allow(dead_code)]
fn calculate_standard_deviation(data: Vec<f64>) -> f64 {
  let mean = calculate_mean(data.clone());
  let variance = calculate_variance(data.clone(), mean);
  variance.sqrt()
}
