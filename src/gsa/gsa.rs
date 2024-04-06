use crate::particle_trait::Mass;
use crate::particle_trait::{Position, Velocity};
use crate::problem;
use crate::pso_trait::{
  Data, DataExporter, GlobalBestPos, Name, OptimizationProblem, ParamValue, ParticleOptimizer, Particles,
};
use crate::rand::Rng;
use crate::utils;
use crate::GSAParticle;
use nalgebra::DVector;
use problem::Problem;
use crate::particle_trait::BestPosition;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::mem;
use std::path::PathBuf;

#[derive(Clone)]
pub struct GSA<GSAParticle> {
  name: String,
  problem: Problem,
  particles: Vec<GSAParticle>,
  global_best_pos: Option<DVector<f64>>,
  influences: Vec<bool>,
  g: f64,
  data: Vec<(f64, Vec<GSAParticle>)>,
  out_directory: PathBuf,
  g0: f64,
  alpha: f64,
}

impl ParticleOptimizer<GSAParticle> for GSA<GSAParticle> {
  fn new(
    name: &str,
    problem: Problem,
    parameters: HashMap<String, ParamValue>,
    out_directory: PathBuf,
  ) -> GSA<GSAParticle> {
    assert!(
      parameters.contains_key("particle_count"),
      "Key 'particle_count' not found."
    );
    let number_of_particles: usize;
    match parameters["particle_count"] {
      ParamValue::Int(val) => number_of_particles = (val as usize).try_into().unwrap(),
      _ => {
        eprintln!("Error: parameter 'particle_count' should be of type Param::Int.");
        std::process::exit(1);
      }
    }

    assert!(parameters.contains_key("g0"), "Key 'g0' not found.");
    let g0: f64;
    match parameters["g0"] {
      ParamValue::Float(val) => g0 = val,
      _ => {
        eprintln!("Error: parameter 'g0' should be of type Param::Float.");
        std::process::exit(1);
      }
    }

    assert!(parameters.contains_key("alpha"), "Key 'alpha' not found.");
    let alpha: f64;
    match parameters["alpha"] {
      ParamValue::Float(val) => alpha = val,
      _ => {
        eprintln!("Error: parameter 'alpha' should be of type Param::Float.");
        std::process::exit(1);
      }
    }

    let mut gsa = GSA {
      name: name.to_owned(),
      problem,
      particles: Vec::new(),
      global_best_pos: None,
      influences: vec![false; number_of_particles],
      g: g0,
      data: Vec::new(),
      out_directory,
      g0,
      alpha,
    };

    gsa.init(number_of_particles);
    gsa
  }

  fn init(&mut self, number_of_particles: usize) {
    let problem = &mut self.problem();
    let mut particles: Vec<GSAParticle> = Vec::new();
    for _ in 0..number_of_particles {
      particles.push(GSAParticle::new(problem));
    }

    let mut global_best_pos = None;

    for particle in particles.clone() {
      if global_best_pos.is_none() || problem.f(&particle.pos()) < problem.f(global_best_pos.as_ref().unwrap()) {
        global_best_pos = Some(particle.pos().clone());
      }
    }

    self.particles = particles;
    self.global_best_pos = Some(global_best_pos.unwrap());
    self.add_data();

    utils::create_directory(self.out_directory().to_path_buf(), true, false);
  }

  fn calculate_vel(&mut self, idx: usize) -> DVector<f64> {
    assert!(idx < self.particles().len());

    let mut a: DVector<f64> = DVector::from_element(self.problem().dim(), 0.);

    let mut rng = rand::thread_rng();

    for j in 0..self.particles().len() {
      if idx == j || !self.influences[j] {
        continue;
      }
      let r = self.particles()[j].pos() - self.particles()[idx].pos();
      let mut a_delta = self.g * self.particles()[j].mass() / (r.norm() + std::f64::EPSILON) * r;

      for e in a_delta.iter_mut() {
        let rand: f64 = rng.gen_range(0.0..1.0);
        *e = rand * *e;
      }

      a += a_delta;
    }

    let rand: f64 = rng.gen_range(0.0..1.0);
    rand * self.particles()[idx].vel() + a
  }

  fn run(&mut self, iterations: usize) {
    for iter in 0..iterations {
      self.g = self.g0 * (-self.alpha * iter as f64 / iterations as f64).exp();

      // Calculate M
      let mut m_unscaled = Vec::new();
      let mut m_sum = 0.;
      let mut best = None;
      let mut worst = None;
      for idx in 0..self.particles().len() {
        let pos = self.particles()[idx].pos().clone();
        if best.is_none() || self.problem().f(&pos) < self.problem().f(&best.as_ref().unwrap()) {
          best = Some(pos.clone());
        }
        if worst.is_none() || self.problem().f(&pos) > self.problem().f(&worst.as_ref().unwrap()) {
          worst = Some(pos.clone());
        }
      }

      let mut m: Vec<f64> = Vec::new();
      if best == worst {
        // This is for when all of the particles are at the exact same position.
        // It can happen during grid search when values for g0 are weird.
        m = vec![0.; self.problem().dim()];
      } else {
        for idx in 0..self.particles().len() {
          let p = self.particles()[idx].clone();
          let numerator = self.problem().f(p.pos()) - self.problem().f(&worst.as_ref().unwrap());
          let denominator = self.problem().f(&best.as_ref().unwrap()) - self.problem().f(&worst.as_ref().unwrap());
          assert!(
            numerator <= 0.,
            "Numerator must be less than or equal to 0: {}",
            numerator
          );
          assert!(denominator < 0., "Denominator must be less than 0: {}", denominator);
          let m_i = numerator / denominator;
          m_unscaled.push(m_i);
          m_sum += m_i;
        }

        for idx in 0..self.particles().len() {
          m.push(m_unscaled[idx] / m_sum);
        }

        // Only use k largest values. Set others to 0.
        let mut m_sorted = m.clone();
        m_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let particle_count = self.particles().len();
        let mut k = (-(particle_count as f64) / (iterations as f64) * iter as f64 + particle_count as f64) as usize;
        k = std::cmp::max(k, 1);
        k = std::cmp::min(k, particle_count);

        for i in 0..particle_count {
          let loc;
          match m_sorted.binary_search_by(|v| v.partial_cmp(&m[i]).expect("Couldn't compare values")) {
            Ok(val) => loc = val,
            Err(val) => loc = val,
          }
          if (particle_count - loc) > k {
            self.influences[i] = false;
          } else {
            self.influences[i] = true;
          }
        }
      }
      for (idx, particle) in self.particles_mut().iter_mut().enumerate() {
        particle.set_mass(m[idx]);
      }

      // Calculate vels.
      let mut vels = Vec::new();
      for idx in 0..self.particles().len() {
        vels.push(self.calculate_vel(idx));
      }

      // Clear memory.
      self.problem().clear_memo();

      // Update the position, best and worst.
      let mut new_global_best_pos = self.global_best_pos.clone().unwrap();
      for idx in 0..self.particles().len() {
        let mut temp_problem = mem::replace(&mut self.problem, Problem::default());
        let particle = &mut self.particles_mut()[idx];
        particle.set_vel(vels[idx].clone());
        let _ = particle.move_pos(&mut temp_problem);
        let _ = particle.update_best_pos(&mut temp_problem);
        let pos = particle.pos().clone();
        if self.problem().f(&pos) < self.problem().f(&new_global_best_pos) {
          new_global_best_pos = self.particles()[idx].pos().clone();
        }
        self.problem = temp_problem;
      }
      self.global_best_pos = Some(new_global_best_pos);

      // Save the data for current iteration.
      self.add_data();
    }
  }
}

impl<T> Particles<T> for GSA<T> {
  fn particles(&self) -> &Vec<T> {
    &self.particles
  }

  fn particles_mut(&mut self) -> &mut Vec<T> {
    &mut self.particles
  }
}

impl<T> GlobalBestPos for GSA<T> {
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

impl<T> OptimizationProblem for GSA<T> {
  fn problem(&mut self) -> &mut Problem {
    &mut self.problem
  }
}

impl<T> Name for GSA<T> {
  fn name(&self) -> &String {
    &self.name
  }
}

impl<T: Clone> Data<T> for GSA<T> {
  fn data(&self) -> &Vec<(f64, Vec<T>)> {
    &self.data
  }

  fn add_data(&mut self) {
    let gbest = self.problem.f(&self.global_best_pos());
    let particles = self.particles.clone();
    self.data.push((gbest, particles));
  }
}

impl<T: Position + Velocity + Mass + Clone> DataExporter<T> for GSA<T> {
  fn out_directory(&self) -> &PathBuf {
    &self.out_directory
  }

  fn save_data(&mut self) -> Result<(), Box<dyn std::error::Error>> {
    // Serialize it to a JSON string
    let mut vec_data = Vec::new();
    for t in 0..self.data().len() {
      let mut iter_data = Vec::new();
      for i in 0..self.data()[t].1.len() {
        let pos = self.data()[t].1[i].pos().clone();
        iter_data.push(json!({
          "fitness": self.problem().f_no_memo(&pos),
          "vel": self.data()[t].1[i].vel().as_slice(),
          "pos": self.data()[t].1[i].pos().as_slice(),
          "mass": self.data()[t].1[i].mass(),
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
