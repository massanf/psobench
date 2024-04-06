use crate::particle_trait::Mass;
use crate::particle_trait::{Position, Velocity};
use crate::problem;
use crate::pso_trait::{
  Data, DataExporter, GlobalBestPos, Name, OptimizationProblem, ParamValue, ParticleOptimizer, Particles,
};
use crate::utils;
use crate::FDOParticle;
use nalgebra::DVector;
use problem::Problem;
use serde_json::json;
use std::{collections::HashMap, fs, mem, path::PathBuf};

#[derive(Clone)]
pub struct FDO<FDOParticle> {
  name: String,
  problem: Problem,
  particles: Vec<FDOParticle>,
  global_best_pos: Option<DVector<f64>>,
  data: Vec<(f64, Vec<FDOParticle>)>,
  out_directory: PathBuf,
  wf: bool,
}

impl ParticleOptimizer<FDOParticle> for FDO<FDOParticle> {
  fn new(
    name: &str,
    problem: Problem,
    parameters: HashMap<String, ParamValue>,
    out_directory: PathBuf,
  ) -> FDO<FDOParticle> {
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

    assert!(parameters.contains_key("wf"), "Key 'wf' not found.");
    let wf: bool;
    match parameters["wf"] {
      ParamValue::Int(val) => match val {
        0 => wf = false,
        1 => wf = true,
        _ => {
          eprintln!("Error: parameter 'wf' must be either 0 or 1.");
          std::process::exit(1);
        }
      },
      _ => {
        eprintln!("Error: parameter 'wf' should be of type Param::Int.");
        std::process::exit(1);
      }
    }

    let mut gsa = FDO {
      name: name.to_owned(),
      problem,
      particles: Vec::new(),
      global_best_pos: None,
      data: Vec::new(),
      out_directory,
      wf,
    };

    gsa.init(number_of_particles);
    gsa
  }

  fn init(&mut self, number_of_particles: usize) {
    let problem = &mut self.problem();
    let mut particles: Vec<FDOParticle> = Vec::new();
    for _ in 0..number_of_particles {
      particles.push(FDOParticle::new(problem));
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
    self.particles()[idx].vel().clone()
  }

  fn run(&mut self, iterations: usize) {
    for _ in 0..iterations {
      // Move problem.
      let mut problem = mem::replace(&mut self.problem, Problem::default());

      let wf = self.wf.clone();
      let mut new_global_best_pos = None;
      for particle in self.particles() {
        if new_global_best_pos.is_none() || problem.f(particle.pos()) < problem.f(new_global_best_pos.clone().unwrap())
        {
          new_global_best_pos = Some(particle.pos());
        }
      }

      self.set_global_best_pos(new_global_best_pos.clone().unwrap().clone());
      let global_best_pos = &self.global_best_pos().clone();
      let f_global_best = self.problem().f(&global_best_pos);

      for particle in self.particles_mut() {
        let pos = particle.pos().clone();
        let vel = particle.vel().clone();
        let r = utils::uniform_distribution(
          &DVector::from_element(global_best_pos.len(), -1.),
          &DVector::from_element(global_best_pos.len(), 1.),
        );

        let fw;
        if problem.f(&pos) == 0. {
          fw = f64::NAN;
        } else {
          if wf {
            fw = (f_global_best / problem.f(&pos)).abs() - 1.;
          } else {
            fw = (f_global_best / problem.f(&pos)).abs();
          }
        }
        let mut new_vel = DVector::from_element(global_best_pos.len(), 0.);
        for d in 0..global_best_pos.len() {
          if fw == 1. || fw.is_nan() {
            // TODO: confused; I don't think this is correct.
            new_vel[d] = pos.clone()[d] * r[d];
          } else if fw == 0. {
            new_vel[d] = (global_best_pos.clone() - pos.clone())[d] * r[d];
          } else {
            if r[d] >= 0. {
              new_vel[d] = (pos.clone() - global_best_pos.clone())[d] * fw;
            } else {
              new_vel[d] = (pos.clone() - global_best_pos.clone())[d] * fw * -1.;
            }
          }
        }

        let new_pos = pos.clone() + new_vel.clone();
        if problem.f(&new_pos) < problem.f(&pos) {
          particle.set_vel(new_vel);
          particle.move_pos(&mut problem);
        } else {
          let new_pos = pos.clone() + vel.clone();
          if problem.f(&new_pos) < problem.f(&pos) {
            particle.move_pos(&mut problem);
          } else {
            // Do nothing.
          }
        }
      }

      // Clear memory.
      self.problem().clear_memo();

      // Set problem.
      self.problem = problem;

      // Save the data for current iteration.
      self.add_data();
    }
  }
}

impl<T> Particles<T> for FDO<T> {
  fn particles(&self) -> &Vec<T> {
    &self.particles
  }

  fn particles_mut(&mut self) -> &mut Vec<T> {
    &mut self.particles
  }
}

impl<T> GlobalBestPos for FDO<T> {
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

impl<T> OptimizationProblem for FDO<T> {
  fn problem(&mut self) -> &mut Problem {
    &mut self.problem
  }
}

impl<T> Name for FDO<T> {
  fn name(&self) -> &String {
    &self.name
  }
}

impl<T: Clone> Data<T> for FDO<T> {
  fn data(&self) -> &Vec<(f64, Vec<T>)> {
    &self.data
  }

  fn add_data(&mut self) {
    let gbest = self.problem.f(&self.global_best_pos());
    let particles = self.particles.clone();
    self.data.push((gbest, particles));
  }
}

impl<T: Position + Velocity + Mass + Clone> DataExporter<T> for FDO<T> {
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
