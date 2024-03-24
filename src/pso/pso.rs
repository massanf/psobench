use crate::optimization_problem;
use crate::particle_trait::ParticleTrait;
use crate::pso_trait::PSOTrait;
use crate::pso_trait::ParamValue;
use nalgebra::DVector;
use optimization_problem::Problem;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone)]
pub struct PSO<T: ParticleTrait> {
  name: String,
  problem: Problem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  data: Vec<(f64, Vec<T>)>,
  parameters: HashMap<String, ParamValue>,
  out_directory: PathBuf,
}

impl<T: ParticleTrait> PSOTrait<T> for PSO<T> {
  fn new(name: &str, problem: Problem, parameters: HashMap<String, ParamValue>, out_directory: PathBuf) -> PSO<T> {
    let mut particles: Vec<T> = Vec::new();

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

    for _ in 0..number_of_particles {
      particles.push(ParticleTrait::new(&problem));
    }

    let mut pso = PSO {
      name: name.to_owned(),
      problem,
      particles,
      global_best_pos: None,
      data: Vec::new(),
      parameters,
      out_directory,
    };

    pso.init();
    pso
  }

  fn name(&self) -> &String {
    &self.name
  }

  fn particles(&self) -> &Vec<T> {
    &self.particles
  }

  fn init_particles(&mut self, problem: &Problem) {
    for i in 0..self.particles.len() {
      self.particles[i].init(problem);
    }
  }

  fn problem(&self) -> &Problem {
    &self.problem
  }

  fn parameters(&self) -> &HashMap<String, ParamValue> {
    &self.parameters
  }

  fn out_directory(&self) -> &PathBuf {
    &self.out_directory
  }

  fn global_best_pos(&self) -> DVector<f64> {
    self.global_best_pos.clone().unwrap()
  }

  fn set_global_best_pos(&mut self, pos: DVector<f64>) {
    self.global_best_pos = Some(pos);
  }

  fn option_global_best_pos(&self) -> &Option<DVector<f64>> {
    &self.global_best_pos
  }

  fn data(&self) -> &Vec<(f64, Vec<T>)> {
    &self.data
  }

  fn add_data(&mut self) {
    let gbest = self.problem().f(&self.global_best_pos());
    let particles = self.particles().clone();
    self.data.push((gbest, particles));
  }

  fn run(&mut self, iterations: usize) {
    for _ in 0..iterations {
      let global_best_pos = self.global_best_pos();
      let mut new_global_best_pos = self.global_best_pos().clone();
      let params = self.parameters().clone();
      for particle in &mut self.particles {
        particle.update_vel(&global_best_pos, &self.problem, &params);
        if particle.update_pos(&self.problem) {
          if self.problem.f(&particle.best_pos()) < self.problem.f(&new_global_best_pos) {
            new_global_best_pos = particle.best_pos().clone();
          }
        }
      }
      self.global_best_pos = Some(new_global_best_pos);

      // Save the data for current iteration.
      self.add_data();
    }
  }
}
