use crate::optimizers::traits::{
  Data, DataExporter, GlobalBestPos, Name, OptimizationProblem, Optimizer, ParamValue, Particles,
};
use crate::particles::traits::{Behavior, Edge, Mass, Particle, Position, Velocity};
use crate::problems;
use crate::rand::Rng;
use crate::utils;
use nalgebra::DVector;
use problems::Problem;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::mem;
use std::path::PathBuf;
use strum_macros::EnumIter;

#[allow(dead_code)]
#[derive(Clone, Copy, EnumIter, Debug, PartialEq)]
pub enum Normalizer {
  MinMax,
  Sigmoid,
  Softmax,
  Rank,
}

#[derive(Clone)]
pub struct Gsa<T> {
  name: String,
  problem: Problem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  influences: Vec<bool>,
  g: f64,
  data: Vec<(f64, Option<Vec<T>>)>,
  out_directory: PathBuf,
  g0: f64,
  alpha: f64,
  save: bool,
  normalizer: Normalizer,
  tiled: bool,
}

impl<T: Particle + Position + Velocity + Mass + Clone> Optimizer<T> for Gsa<T> {
  fn new(
    name: String,
    problem: Problem,
    parameters: HashMap<String, ParamValue>,
    out_directory: PathBuf,
    behavior: Behavior,
    save: bool,
  ) -> Gsa<T> {
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

    assert!(parameters.contains_key("tiled"), "Key 'tiled' not found.");
    let tiled = match parameters["tiled"] {
      ParamValue::Tiled(val) => val,
      _ => {
        eprintln!("Error: parameter 'tiled' should be of type Param::Tiled.");
        std::process::exit(1);
      }
    };

    if (tiled && behavior.edge != Edge::Cycle) || (!tiled && behavior.edge == Edge::Cycle) {
      let name = match tiled {
        true => "tiled",
        false => "untiled",
      };
      eprintln!("Running {} but edge behavior is {:?}", name, behavior.edge);
    }

    let mut gsa = Gsa {
      name,
      problem,
      particles: Vec::new(),
      global_best_pos: None,
      influences: vec![false; number_of_particles],
      g: g0,
      data: Vec::new(),
      out_directory,
      g0,
      alpha,
      save,
      normalizer,
      tiled,
    };

    gsa.init(number_of_particles, behavior);
    gsa
  }

  fn init(&mut self, number_of_particles: usize, behavior: Behavior) {
    let problem = &mut self.problem();
    let mut particles: Vec<T> = Vec::new();
    for _ in 0..number_of_particles {
      particles.push(T::new(problem, behavior));
    }

    let mut global_best_pos = None;
    for particle in particles.clone() {
      if global_best_pos.is_none() || problem.f(particle.pos()) < problem.f(global_best_pos.as_ref().unwrap()) {
        global_best_pos = Some(particle.pos().clone());
      }
    }

    self.particles = particles;
    self.set_global_best_pos(global_best_pos.unwrap());

    utils::create_directory(self.out_directory().to_path_buf(), true, false);
  }

  fn calculate_vel(&mut self, i: usize) -> DVector<f64> {
    assert!(i < self.particles().len());
    let mut a: DVector<f64> = DVector::from_element(self.problem().dim(), 0.);
    let mut rng = rand::thread_rng();

    for j in 0..self.particles().len() {
      if i == j || !self.influences[j] {
        continue;
      }

      let i = self.particles()[i].clone();
      let j = self.particles()[j].clone();

      let r = match self.tiled {
        true => {
          let width = self.problem().domain().1 - self.problem().domain().0;
          let mut closest_j = j.pos().clone();

          for (idx, x) in closest_j.iter_mut().enumerate() {
            if (*x - width - i.pos()[idx]).abs() < (*x - i.pos()[idx]).abs() {
              *x -= width;
            } else if (*x + width - i.pos()[idx]).abs() < (*x - i.pos()[idx]).abs() {
              *x += width;
            }
          }
          closest_j - i.pos().clone()
        }
        false => j.pos() - i.pos(),
      };

      let mut a_delta = self.g * j.mass() / (r.norm() + std::f64::EPSILON) * r;

      for e in a_delta.iter_mut() {
        let rand: f64 = rng.gen_range(0.0..1.0);
        *e *= rand;
      }

      a += a_delta;
    }

    let rand: f64 = rng.gen_range(0.0..1.0);
    rand * self.particles()[i].vel() + a
  }

  fn run(&mut self, iterations: usize) {
    for iter in 0..iterations {
      self.g = self.g0 * (-self.alpha * iter as f64 / iterations as f64).exp();

      let mut fitness = Vec::new();
      for idx in 0..self.particles().len() {
        let pos = self.particles()[idx].pos().clone();
        fitness.push(self.problem().f(&pos));
      }

      let m = match self.normalizer {
        Normalizer::MinMax => utils::original_gsa_normalize(fitness),
        Normalizer::Sigmoid => utils::sigmoid_normalize(fitness),
        Normalizer::Softmax => utils::softmax_normalize(fitness),
        Normalizer::Rank => utils::rank_normalize(fitness),
      };
      for (mass, particle) in m.iter().zip(self.particles_mut().iter_mut()) {
        particle.set_mass(*mass);
      }

      // Only use k largest values. Make others not influence.
      let mut m_sorted = m.clone();
      m_sorted.sort_by(|a, b| a.partial_cmp(b).expect("Could not compare NaN."));

      let particle_count = self.particles().len();
      let mut k = (-(particle_count as f64) / (iterations as f64) * iter as f64 + particle_count as f64) as usize;
      k = std::cmp::max(k, 1);
      k = std::cmp::min(k, particle_count);

      for (i, m_i) in m.iter().enumerate().take(particle_count) {
        let loc = match m_sorted.binary_search_by(|v| v.partial_cmp(m_i).expect("Couldn't compare values")) {
          Ok(val) => val,
          Err(val) => val,
        };
        self.influences[i] = (particle_count - loc) <= k;
      }

      // Calculate vels.
      let mut vels = Vec::new();
      for idx in 0..self.particles().len() {
        vels.push(self.calculate_vel(idx));
      }

      // Clear memory.
      self.problem().clear_memo();

      // Update the position, best and worst.
      let mut new_global_best_pos = None;
      // for (i, m_i) in 0..self.particles().len() {
      for (i, vel) in vels.iter().enumerate().take(self.particles().len()) {
        let mut temp_problem = mem::take(&mut self.problem);
        let particle = &mut self.particles_mut()[i];
        particle.update_vel(vel.clone(), &mut temp_problem);
        particle.move_pos(&mut temp_problem);
        let pos = particle.pos().clone();
        if new_global_best_pos.is_none()
          || (self.problem().f(&pos) < self.problem().f(&new_global_best_pos.clone().unwrap()))
        {
          new_global_best_pos = Some(pos);
        }
        self.problem = temp_problem;
      }
      self.update_global_best_pos(new_global_best_pos.unwrap());

      // Save the data for current iteration.
      let gbest = self.problem.f(&self.global_best_pos());
      let particles = self.particles.clone();
      self.add_data(self.save, gbest, particles);
    }
  }
}

impl<T> Particles<T> for Gsa<T> {
  fn particles(&self) -> &Vec<T> {
    &self.particles
  }

  fn particles_mut(&mut self) -> &mut Vec<T> {
    &mut self.particles
  }
}

impl<T> GlobalBestPos for Gsa<T> {
  fn global_best_pos(&self) -> DVector<f64> {
    self.global_best_pos.clone().unwrap()
  }

  fn option_global_best_pos(&self) -> &Option<DVector<f64>> {
    &self.global_best_pos
  }

  fn set_global_best_pos(&mut self, pos: DVector<f64>) {
    self.global_best_pos = Some(pos);
  }
}

impl<T> OptimizationProblem for Gsa<T> {
  fn problem(&mut self) -> &mut Problem {
    &mut self.problem
  }
}

impl<T> Name for Gsa<T> {
  fn name(&self) -> &String {
    &self.name
  }
}

impl<T: Clone> Data<T> for Gsa<T> {
  fn data(&self) -> &Vec<(f64, Option<Vec<T>>)> {
    &self.data
  }

  fn add_data_impl(&mut self, datum: (f64, Option<Vec<T>>)) {
    self.data.push(datum);
  }
}

impl<T: Position + Velocity + Mass + Clone> DataExporter<T> for Gsa<T> {
  fn out_directory(&self) -> &PathBuf {
    &self.out_directory
  }

  fn save_data(&mut self) -> Result<(), Box<dyn std::error::Error>> {
    // Serialize it to a JSON string
    let mut vec_data = Vec::new();
    for t in 0..self.data().len() {
      let mut iter_data = Vec::new();
      let datum = self.data()[t].1.clone().unwrap();
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
        "particles": iter_data
      }));
    }

    let serialized = serde_json::to_string(&json!(vec_data))?;

    fs::write(self.out_directory().join("data.json"), serialized)?;
    Ok(())
  }
}
