use crate::particle_trait;
use crate::problem;
use nalgebra::DVector;
use particle_trait::ParticleTrait;
use problem::Problem;
use serde::ser::{Serialize, Serializer};
use serde_json::json;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::PathBuf;

#[derive(Clone)]
pub enum ParamValue {
  Float(f64),
  Int(isize),
}

impl fmt::Display for ParamValue {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      ParamValue::Float(n) => write!(f, "{:.2}", n),
      ParamValue::Int(c) => write!(f, "{}", c),
    }
  }
}

impl Serialize for ParamValue {
  fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    match *self {
      ParamValue::Float(ref n) => serializer.serialize_f64(*n),
      ParamValue::Int(ref c) => serializer.serialize_u64(*c as u64),
    }
  }
}

pub trait PSOTrait<T: ParticleTrait> {
  fn new(name: &str, problem: Problem, parameters: HashMap<String, ParamValue>, out_directory: PathBuf) -> Self
  where
    Self: Sized;

  fn init(&mut self, number_of_particles: usize);

  fn name(&self) -> &String;

  fn particles(&self) -> &Vec<T>;
  fn particles_mut(&mut self) -> &mut Vec<T>;

  fn calculate_vel(&mut self, idx: usize) -> DVector<f64>;

  fn problem(&mut self) -> &mut Problem;

  fn out_directory(&self) -> &PathBuf;

  fn global_best_pos(&self) -> DVector<f64>;
  fn option_global_best_pos(&self) -> &Option<DVector<f64>>;

  fn data(&self) -> &Vec<(f64, Vec<T>)>;
  fn add_data(&mut self);

  fn run(&mut self, iterations: usize);
  fn experiment(&mut self, trials: usize, iterations: usize) {
    for _ in 0..trials {
      self.run(iterations);
    }
  }

  fn save_data(&mut self) -> Result<(), Box<dyn std::error::Error>> {
    // Serialize it to a JSON string
    let mut vec_data = Vec::new();
    let problem = self.problem().clone();
    for t in 0..self.data().len() {
      let mut iter_data = Vec::new();
      for i in 0..self.data()[t].1.len() {
        let pos = self.data()[t].1[i].pos().clone();
        iter_data.push(json!({
          "fitness": problem.clone().f(&pos),
          "vel": self.data()[t].1[i].vel().as_slice(),
          "pos": self.data()[t].1[i].pos().as_slice(),
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

  fn save_config(&mut self, parameters: &HashMap<String, ParamValue>) -> Result<(), Box<dyn std::error::Error>> {
    let serialized = serde_json::to_string(&json!({
      "problem": {
        "name": self.problem().name(),
        "dim": self.problem().dim(),
    },
      "method": {
        "name": self.name(),
        "parameters": parameters,
      },
    }))?;
    fs::write(self.out_directory().join("config.json"), serialized)?;
    Ok(())
  }

  fn save_summary(&self) -> Result<(), Box<dyn std::error::Error>> {
    let mut global_best_progress = Vec::new();
    for t in 0..self.data().len() {
      global_best_progress.push(self.data()[t].0);
    }
    let serialized = serde_json::to_string(&json!({
      "global_best_fitness": global_best_progress,
    }))?;
    fs::write(self.out_directory().join("summary.json"), serialized)?;
    Ok(())
  }
}
