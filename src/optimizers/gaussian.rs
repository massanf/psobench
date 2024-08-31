use crate::optimizers::traits::{
  Data, DataExporter, GlobalBestPos, Name, OptimizationProblem, Optimizer, ParamValue, Particles,
};
use crate::particles::traits::{Behavior, Particle, Position, Velocity};
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

#[derive(Clone)]
pub struct Gaussian<T> {
  name: String,
  problem: Problem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  g: f64,
  data: Vec<(f64, Option<Vec<T>>)>,
  out_directory: PathBuf,
  fitness: Vec<f64>,
  g0: f64,
  alpha: f64,
  save: bool,
}

impl<T: Particle + Position + Velocity + Clone> Optimizer<T> for Gaussian<T> {
  fn new(
    name: String,
    problem: Problem,
    parameters: HashMap<String, ParamValue>,
    out_directory: PathBuf,
    save: bool,
  ) -> Gaussian<T> {
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

    assert!(parameters.contains_key("behavior"), "Key 'behavior' not found.");
    let behavior = match parameters["behavior"] {
      ParamValue::Behavior(val) => val,
      _ => {
        eprintln!("Error: parameter 'behavior' should be of type Param::Behavior.");
        std::process::exit(1);
      }
    };

    let mut gaussian = Gaussian {
      name,
      problem,
      particles: Vec::new(),
      global_best_pos: None,
      g: g0,
      data: Vec::new(),
      out_directory,
      fitness: Vec::new(),
      g0,
      alpha,
      save,
    };

    gaussian.init(number_of_particles, behavior);
    gaussian
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

  fn calculate_vel(&mut self, _i: usize) -> DVector<f64> {
    panic!("`calculate_vel` is left for legacy reasons. Use `calculate_vels`.");
  }

  fn run(&mut self, iterations: usize) {
    for iter in 0..iterations {
      self.g = self.g0 * (-self.alpha * iter as f64 / iterations as f64).exp();

      // Calculate vels.
      let mut temp_problem = mem::take(&mut self.problem);
      let vels = calculate_vels(self.particles().clone(), &mut temp_problem, self.g);
      self.problem = temp_problem;

      // Clear memory.
      self.problem().clear_memo();

      // Update the position, best and worst.
      let mut new_global_best_pos = None;
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

fn calculate_vels<T: Particle + Position + Velocity + Clone>(
  particles: Vec<T>,
  problem: &mut Problem,
  g: f64,
) -> Vec<DVector<f64>> {
  // Calculate fitness for reuse.
  let mut fitnesses = Vec::new();
  for idx in 0..particles.len() {
    let pos = particles[idx].pos().clone();
    fitnesses.push(problem.f(&pos));
  }

  let mut vels = Vec::new();
  for i in 0..particles.len() {
    let mut a: DVector<f64> = DVector::from_element(problem.dim(), 0.);
    let mut rng = rand::thread_rng();

    for j in 0..particles.len() {
      let particle_i = particles[i].clone();
      let particle_j = particles[j].clone();

      let r = particle_j.pos() - particle_i.pos();

      let mut a_delta = g * fitnesses[j] / (r.norm() + std::f64::EPSILON) * r;

      for e in a_delta.iter_mut() {
        let rand: f64 = rng.gen_range(0.0..1.0);
        *e *= rand;
      }

      a += a_delta;
    }

    let rand: f64 = rng.gen_range(0.0..1.0);
    vels.push(rand * particles[i].vel() + a);
  }
  vels
}

impl<T> Particles<T> for Gaussian<T> {
  fn particles(&self) -> &Vec<T> {
    &self.particles
  }

  fn particles_mut(&mut self) -> &mut Vec<T> {
    &mut self.particles
  }
}

impl<T> GlobalBestPos for Gaussian<T> {
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

impl<T> OptimizationProblem for Gaussian<T> {
  fn problem(&mut self) -> &mut Problem {
    &mut self.problem
  }
}

impl<T> Name for Gaussian<T> {
  fn name(&self) -> &String {
    &self.name
  }
}

impl<T: Clone> Data<T> for Gaussian<T> {
  fn data(&self) -> &Vec<(f64, Option<Vec<T>>)> {
    &self.data
  }

  fn add_data_impl(&mut self, datum: (f64, Option<Vec<T>>)) {
    self.data.push(datum);
  }
}

impl<T: Position + Velocity + Clone> DataExporter<T> for Gaussian<T> {
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
