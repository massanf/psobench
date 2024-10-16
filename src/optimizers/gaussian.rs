use crate::optimizers::traits::{
  Data, DataExporter, GlobalBestPos, Name, OptimizationProblem, Optimizer, ParamValue, Particles,
};
use crate::particles::traits::{Behavior, Particle, Position, Velocity};
use crate::problems;
use crate::utils;
use nalgebra::DVector;
use problems::Problem;
use rug::Float;
use serde_json::json;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fs;
use std::mem;
use std::path::PathBuf;

#[derive(Clone)]
pub struct Gaussian<T> {
  name: String,
  problem: Problem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  data: Vec<(f64, Option<Vec<T>>)>,
  out_directory: PathBuf,
  x: Vec<Vec<DVector<f64>>>,
  gamma: f64,
  beta: f64,
  scale: f64,
  fitness: Vec<Vec<f64>>, // f[j(time)][i(particle)]
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

    assert!(parameters.contains_key("behavior"), "Key 'behavior' not found.");
    let behavior = match parameters["behavior"] {
      ParamValue::Behavior(val) => val,
      _ => {
        eprintln!("Error: parameter 'behavior' should be of type Param::Behavior.");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("gamma"), "Key 'gamma' not found.");
    let gamma = match parameters["gamma"] {
      ParamValue::Float(val) => val,
      _ => {
        eprintln!("Error: parameter 'gamma' should be of type Param::Float.");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("beta"), "Key 'beta' not found.");
    let beta = match parameters["beta"] {
      ParamValue::Float(val) => val,
      _ => {
        eprintln!("Error: parameter 'beta' should be of type Param::Float.");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("scale"), "Key 'scale' not found.");
    let scale = match parameters["scale"] {
      ParamValue::Float(val) => val,
      _ => {
        eprintln!("Error: parameter 'scale' should be of type Param::Float.");
        std::process::exit(1);
      }
    };

    let mut gaussian = Gaussian {
      name,
      problem,
      particles: Vec::new(),
      global_best_pos: None,
      data: Vec::new(),
      out_directory,
      x: Vec::new(),
      beta,
      gamma,
      scale,
      fitness: Vec::new(),
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
    panic!("`calculate_vel` is left here for legacy reasons. Use `calculate_vels`.");
  }

  fn run(&mut self, iterations: usize) {
    for _iter in 0..iterations {
      // Calculate and record fitness.
      let mut f = Vec::new();
      let mut x = Vec::new();
      for idx in 0..self.particles.len() {
        let pos = self.particles[idx].pos().clone();
        f.push(self.problem.f(&pos));
        x.push(pos.clone());
      }

      // Update best pos.
      let mut new_global_best_pos = None;
      let mut new_global_best = None;
      for idx in 0..self.particles().len() {
        if new_global_best.is_none() || new_global_best.unwrap() > f[idx] {
          new_global_best = Some(f[idx]);
          new_global_best_pos = Some(self.particles[idx].pos());
        }
      }
      println!("{:?}", new_global_best.unwrap() as i32);
      self.update_global_best_pos(new_global_best_pos.unwrap().clone());

      // Save positions and fitnesses.
      self.fitness.push(f.clone());
      self.x.push(x);

      // Calculate vels.
      let vels = calculate_vels(self.x.clone(), self.fitness.clone(), self.gamma, self.beta, self.scale);

      // Clear memory.
      self.problem().clear_memo();

      // Update the position, best and worst.
      for (i, vel) in vels.iter().enumerate().take(self.particles().len()) {
        let mut temp_problem = mem::take(&mut self.problem);
        let particle = &mut self.particles_mut()[i];
        particle.update_vel(vel.clone(), &mut temp_problem);
        particle.move_pos(&mut temp_problem);
        self.problem = temp_problem;
      }

      // Save the data for current iteration.
      let gbest = self.problem.f(&self.global_best_pos());
      let particles = self.particles.clone();
      self.add_data(self.save, gbest, particles);
    }
  }
}

fn calculate_vels(x: Vec<Vec<DVector<f64>>>, f: Vec<Vec<f64>>, gamma: f64, beta: f64, scale: f64) -> Vec<DVector<f64>> {
  let precision: u32 = 128;
  let t = f.len();
  let n = x[0].len();
  let d = x[0][0].len();

  let mut vels = Vec::with_capacity(n);
  for r in 0..n {
    let mut sum: DVector<Float> = DVector::from_element(d, Float::with_val(precision, 0.));

    let x_t_1_r: DVector<Float> = x[t - 1][r].map(|x| Float::with_val(precision, x));
    for j in std::cmp::max(0, t - 10)..t {
      // for j in 0..t {
      let mut numerator: DVector<Float> = DVector::from_element(d, Float::with_val(precision, 0.));
      let mut denominator: Float = Float::with_val(precision, 0.);
      for i in 0..n {
        if j == t - 1 && i == r {
          continue;
        }
        let x_ji: DVector<Float> = x[j][i].map(|x| Float::with_val(precision, x));
        let f_ji: Float = Float::with_val(precision, f[j][i]);
        let exponent = Float::with_val(
          precision,
          -beta / 2.0 * ((&x[t - 1][r] - &x[j][i]) / (scale * scale)).norm_squared(),
        );
        let g = alpha(beta, d, scale) * (exponent.clone()).exp();
        numerator += (x_ji - x_t_1_r.clone()) * (1. / f_ji.clone() * g.clone() * beta / scale);
        denominator += 1. / f_ji.clone() * g;
      }
      if denominator == 0. {
        panic!("Division by 0");
      }
      sum += numerator / denominator;
    }

    if sum.map(|x| x.to_f64()).norm() > 10000. {
      panic!("Velocity overflow.");
    }

    vels.push(gamma * sum.map(|x| x.to_f64()) * scale);
  }
  println!(
    "{}: avg vel",
    vels.iter().map(|v| v.norm()).sum::<f64>() / vels.len() as f64
  );
  vels
}

fn alpha(beta: f64, d: usize, scale: f64) -> f64 {
  (beta / (2.0 * PI * scale * scale)).powf(d as f64 / 2.0)
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
