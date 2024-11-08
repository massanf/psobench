use crate::optimizers::gsa::Normalizer;
use crate::particles::traits::{Behavior, Position, Velocity};
use crate::problems;
use nalgebra::DVector;
use problems::Problem;
use serde::ser::{Serialize, Serializer};
use serde_json::{json, Map};
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::PathBuf;

#[allow(dead_code)]
#[derive(Clone)]
pub enum ParamValue {
  Float(f64),
  Int(isize),
  Normalizer(Normalizer),
  Bool(bool),
  Behavior(Behavior),
}

impl fmt::Display for ParamValue {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      ParamValue::Float(n) => write!(f, "{:.2}", n),
      ParamValue::Int(c) => write!(f, "{}", c),
      _ => {
        todo!()
      }
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
      ParamValue::Normalizer(value) => match value {
        Normalizer::MinMax => serializer.serialize_str("MinMax"),
        Normalizer::ZScore => serializer.serialize_str("ZScore"),
        Normalizer::Robust => serializer.serialize_str("Robust"),
        Normalizer::Sigmoid2 => serializer.serialize_str("Sigmoid2"),
        Normalizer::Sigmoid4 => serializer.serialize_str("Sigmoid4"),
        Normalizer::Rank => serializer.serialize_str("Rank"),
      },
      ParamValue::Bool(v) => serializer.serialize_bool(v),
      ParamValue::Behavior(b) => b.serialize(serializer),
    }
  }
}

pub trait Optimizer<U: Position + Velocity + Clone>:
  Name + OptimizationProblem + Particles<U> + DataExporter<U>
{
  fn new(
    name: String,
    problem: Problem,
    parameters: HashMap<String, ParamValue>,
    out_directory: PathBuf,
    save: bool,
  ) -> Self
  where
    Self: Sized;

  fn init(&mut self, number_of_particles: usize, behavior: Behavior);
  fn calculate_vel(&mut self, i: usize) -> DVector<f64>;
  fn run(&mut self, iterations: usize);
}

pub trait Particles<T> {
  fn particles(&self) -> &Vec<T>;
  fn particles_mut(&mut self) -> &mut Vec<T>;
}

pub trait GlobalBestPos: OptimizationProblem {
  fn global_best_pos(&self) -> DVector<f64>;
  fn global_worst_pos(&self) -> DVector<f64>;
  #[allow(dead_code)]
  fn option_global_best_pos(&self) -> &Option<DVector<f64>>;
  #[allow(dead_code)]
  fn option_global_worst_pos(&self) -> &Option<DVector<f64>>;
  fn set_global_best_pos(&mut self, pos: DVector<f64>);
  fn set_global_worst_pos(&mut self, pos: DVector<f64>);
  fn update_global_best_pos(&mut self, pos: DVector<f64>) {
    // let gb = self.global_best_pos().clone();
    // if self.problem().f(&pos) < self.problem().f(&gb) {
    self.set_global_best_pos(pos);
    // }
  }
  fn update_global_worst_pos(&mut self, pos: DVector<f64>) {
    // let gb = self.global_best_pos().clone();
    // if self.problem().f(&pos) < self.problem().f(&gb) {
    self.set_global_worst_pos(pos);
    // }
  }
}

pub trait Name {
  fn name(&self) -> &String;
}

pub trait OptimizationProblem {
  fn problem(&mut self) -> &mut Problem;
}

pub trait Data<T>: OptimizationProblem + GlobalBestPos {
  fn data(&self) -> &Vec<(f64, f64, Option<Vec<T>>)>;
  fn additional_data(&self) -> &Vec<Vec<Vec<(String, f64)>>>;
  fn add_data(&mut self, save: bool, gbest: f64, gworst: f64, particles: Vec<T>) {
    if save {
      self.add_data_impl((gbest, gworst, Some(particles)));
    } else {
      self.add_data_impl((gbest, gworst, None));
    }
  }
  fn add_additional_data(&mut self, save: bool, particles: Vec<Vec<(String, f64)>>) {
    if !save {
      return;
    }
    self.add_additional_data_impl(particles);
  }
  fn add_data_impl(&mut self, datum: (f64, f64, Option<Vec<T>>));
  fn add_additional_data_impl(&mut self, datum: Vec<Vec<(String, f64)>>);
}

pub trait DataExporter<T: Position + Velocity + Clone>: Data<T> + Name + OptimizationProblem {
  fn out_directory(&self) -> &PathBuf;
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

  fn save_additional_data(&mut self) -> Result<(), Box<dyn std::error::Error>> {
    // Serialize it to a JSON string
    let mut vec_data = Vec::new();
    for t in 0..self.data().len() {
      let mut iter_data = Vec::new();
      let datum = self.additional_data()[t].clone();
      for particle_datum in &datum {
        let mut map = Map::new();
        for datum in &particle_datum.clone() {
          map.insert(
            datum.clone().0, datum.clone().1.into()
          );
        }
        iter_data.push(map);
      }
      vec_data.push(json!({
        "particles": iter_data
      }));
    }

    let serialized = serde_json::to_string(&json!(vec_data))?;

    fs::write(self.out_directory().join("additional_data.json"), serialized)?;
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

  fn save_summary(&mut self) -> Result<(), Box<dyn std::error::Error>> {
    let mut global_best_progress = Vec::new();
    let mut global_worst_progress = Vec::new();
    for t in 0..self.data().len() {
      global_best_progress.push(self.data()[t].0);
      global_worst_progress.push(self.data()[t].1);
    }
    let serialized = serde_json::to_string(&json!({
      "global_best_fitness": global_best_progress,
      "global_worst_fitness": global_worst_progress,
      "evaluation_count": self.problem().cnt(),
    }))?;
    fs::write(self.out_directory().join("summary.json"), serialized)?;
    Ok(())
  }
}
