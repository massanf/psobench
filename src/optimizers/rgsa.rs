use crate::optimizers::gsa::Normalizer;
use crate::optimizers::traits::{
  Data, DataExporter, GlobalBestPos, Name, OptimizationProblem, Optimizer, ParamValue, Particles,
};
use crate::particles::traits::{Behavior, Mass, Particle, Position, Velocity};
use crate::problems;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;
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
pub struct Rgsa<T> {
  name: String,
  problem: Problem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  global_worst_pos: Option<DVector<f64>>,
  g: f64,
  data: Vec<(f64, f64, Option<Vec<T>>)>,
  additional_data: Vec<Vec<Vec<(String, f64)>>>,
  out_directory: PathBuf,
  g0: f64,
  alpha: f64,
  save: bool,
  normalizer: Normalizer,
}

impl<T: Particle + Position + Velocity + Mass + Clone> Optimizer<T> for Rgsa<T> {
  fn new(
    name: String,
    problem: Problem,
    parameters: HashMap<String, ParamValue>,
    out_directory: PathBuf,
    save: bool,
  ) -> Rgsa<T> {
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

    let mut rgsa = Rgsa {
      name,
      problem,
      particles: Vec::new(),
      global_best_pos: None,
      global_worst_pos: None,
      g: g0,
      data: Vec::new(),
      additional_data: Vec::new(),
      out_directory,
      g0,
      alpha,
      save,
      normalizer,
    };

    rgsa.init(number_of_particles, behavior);
    rgsa
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

    for iter in 0..iterations {
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
      let use_avg = true;
      let spread = match use_avg {
        true => distances.iter().sum::<f64>() / distances.len() as f64,
        false => utils::calculate_std(&distances),
      };
      if initial_spread.is_none() {
        initial_spread = Some(spread);
      }
      let _spread_ratio = spread / initial_spread.unwrap();

      let ratio = (-self.alpha * iter as f64 / iterations as f64).exp();

      self.g = (-self.alpha * iter as f64 / iterations as f64).exp();

      let mut fitness = Vec::new();
      for idx in 0..n {
        let pos = self.particles()[idx].pos().clone();
        fitness.push(self.problem().f(&pos));
      }

      let m = utils::original_gsa_mass(fitness.clone());

      for (mass, particle) in m.iter().zip(self.particles_mut().iter_mut()) {
        particle.set_mass(*mass);
      }

      // Calculate vels.
      let mut x: Vec<DVector<f64>> = Vec::new();
      let mut v = Vec::new();
      for idx in 0..n {
        x.push(self.particles()[idx].pos().clone());
        v.push(self.particles()[idx].vel().clone());
      }

      let (vels, additional_data) = calculate_vels(
        x.clone(),
        m.clone(),
        self.g,
        iter as f64 / iterations as f64,
        ratio,
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
      self.add_additional_data(self.save, additional_data);
    }
  }
}

fn calculate_vels(
  x: Vec<DVector<f64>>,
  f: Vec<f64>,
  large_g: f64,
  progress: f64,
  _spread: f64,
) -> (Vec<DVector<f64>>, Vec<Vec<(String, f64)>>) {
  let n = x.len();
  let d = x[0].len();

  let elite_count = std::cmp::min(std::cmp::max((n as f64 * (1. - progress as f64)) as usize, 1), n);
  let mut sorted_f = f.clone();
  sorted_f.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
  let elites: Vec<f64> = sorted_f.iter().take(elite_count).copied().collect();
  let influences: Vec<bool> = f.iter().map(|x| elites.contains(x)).collect();

  let mut weighted_sum_x: DVector<f64> = DVector::from_element(d, 0.);
  let mut sum_f = 0.;
  for k in 0..n {
    if !influences[k] {
      continue;
    }
    weighted_sum_x += f[k] * x[k].clone();
    sum_f += f[k];
  }
  let cg = weighted_sum_x / sum_f;
  let std = 50. * large_g;

  let new_x = generate_random_dvectors(&cg, std, n);

  let mut vels = Vec::new();

  for k in 0..n {
    vels.push(new_x[k].clone() - x[k].clone());
  }

  let additional_data = Vec::new();
  (vels, additional_data)
}

fn generate_random_dvectors(cg: &DVector<f64>, std: f64, n: usize) -> Vec<DVector<f64>> {
  let mut rng = rand::thread_rng();
  (0..n)
    .map(|_| {
      DVector::from_iterator(
        cg.len(),
        cg.iter().map(|&mean| {
          let normal = Normal::new(mean, std).unwrap();
          normal.sample(&mut rng)
        }),
      )
    })
    .collect()
}

impl<T> Particles<T> for Rgsa<T> {
  fn particles(&self) -> &Vec<T> {
    &self.particles
  }

  fn particles_mut(&mut self) -> &mut Vec<T> {
    &mut self.particles
  }
}

impl<T> GlobalBestPos for Rgsa<T> {
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

impl<T> OptimizationProblem for Rgsa<T> {
  fn problem(&mut self) -> &mut Problem {
    &mut self.problem
  }
}

impl<T> Name for Rgsa<T> {
  fn name(&self) -> &String {
    &self.name
  }
}

impl<T: Clone> Data<T> for Rgsa<T> {
  fn data(&self) -> &Vec<(f64, f64, Option<Vec<T>>)> {
    &self.data
  }

  fn additional_data(&self) -> &Vec<Vec<Vec<(String, f64)>>> {
    &self.additional_data
  }

  fn add_data_impl(&mut self, datum: (f64, f64, Option<Vec<T>>)) {
    self.data.push(datum);
  }

  fn add_additional_data_impl(&mut self, datum: Vec<Vec<(String, f64)>>) {
    self.additional_data.push(datum);
  }
}

impl<T: Position + Velocity + Mass + Clone> DataExporter<T> for Rgsa<T> {
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
