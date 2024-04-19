use crate::optimizers::traits::{
  Data, DataExporter, GlobalBestPos, Name, OptimizationProblem, Optimizer, ParamValue, Particles,
};
use crate::particles::traits::{Behavior, BestPosition, Particle, Position, Velocity};
use crate::problems;
use crate::rand::Rng;
use crate::utils;
use nalgebra::DVector;
use problems::Problem;
use std::{collections::HashMap, mem, path::PathBuf};

#[derive(Clone)]
pub struct Pso<T> {
  name: String,
  problem: Problem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  data: Vec<(f64, Option<Vec<T>>)>,
  out_directory: PathBuf,
  behavior: Behavior,
  w: f64,
  phi_p: f64,
  phi_g: f64,
  save: bool,
}

impl<T: Particle + Position + Velocity + BestPosition + Clone> Optimizer<T> for Pso<T> {
  fn new(
    name: String,
    problem: Problem,
    parameters: HashMap<String, ParamValue>,
    out_directory: PathBuf,
    behavior: Behavior,
    save: bool,
  ) -> Pso<T> {
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

    assert!(parameters.contains_key("w"), "Key 'w' not found.");
    let w = match parameters["w"] {
      ParamValue::Float(val) => val,
      _ => {
        eprintln!("Error");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("phi_p"), "Key 'phi_p' not found.");
    let phi_p = match parameters["phi_p"] {
      ParamValue::Float(val) => val,
      _ => {
        eprintln!("Error");
        std::process::exit(1);
      }
    };

    assert!(parameters.contains_key("phi_g"), "Key 'phi_g' not found.");
    let phi_g = match parameters["phi_g"] {
      ParamValue::Float(val) => val,
      _ => {
        eprintln!("Error");
        std::process::exit(1);
      }
    };

    let mut pso = Pso {
      name,
      problem,
      particles: Vec::new(),
      global_best_pos: None,
      data: Vec::new(),
      out_directory,
      behavior,
      w,
      phi_p,
      phi_g,
      save,
    };

    pso.init(number_of_particles);
    pso
  }

  fn init(&mut self, number_of_particles: usize) {
    let behavior = self.behavior;
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

    utils::create_directory(self.out_directory().to_path_buf(), false, true);
  }

  fn calculate_vel(&mut self, idx: usize) -> DVector<f64> {
    assert!(idx < self.particles().len());

    let mut rng = rand::thread_rng();
    let r_p: f64 = rng.gen_range(0.0..1.0);
    let r_g: f64 = rng.gen_range(0.0..1.0);

    let mut new_vel = self.w * self.particles()[idx].vel()
      + self.phi_p * r_p * (self.particles()[idx].best_pos() - self.particles()[idx].pos())
      + self.phi_g * r_g * (self.global_best_pos() - self.particles()[idx].pos());
    for e in new_vel.iter_mut() {
      if *e > self.problem().domain().1 - self.problem().domain().0 {
        *e = self.problem().domain().1 - self.problem().domain().0;
      } else if *e < self.problem().domain().0 - self.problem().domain().1 {
        *e = self.problem().domain().0 - self.problem().domain().1;
      }
    }

    new_vel
  }

  fn run(&mut self, iterations: usize) {
    for _ in 0..iterations {
      self.problem().clear_memo();

      let mut new_global_best_pos = None;
      for idx in 0..self.particles().len() {
        let vel = self.calculate_vel(idx);
        let mut temp_problem = mem::take(&mut self.problem);
        let particle = &mut self.particles_mut()[idx];
        particle.update_vel(vel, &mut temp_problem);
        particle.move_pos(&mut temp_problem);
        particle.update_best_pos(&mut temp_problem);
        let best_pos = self.particles()[idx].best_pos().clone();
        if new_global_best_pos.is_none()
          || self.problem().f(&best_pos) < self.problem.f(&new_global_best_pos.clone().unwrap())
        {
          new_global_best_pos = Some(self.particles()[idx].best_pos().clone());
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

impl<T> Particles<T> for Pso<T> {
  fn particles(&self) -> &Vec<T> {
    &self.particles
  }

  fn particles_mut(&mut self) -> &mut Vec<T> {
    &mut self.particles
  }
}

impl<T> GlobalBestPos for Pso<T> {
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

impl<T> OptimizationProblem for Pso<T> {
  fn problem(&mut self) -> &mut Problem {
    &mut self.problem
  }
}

impl<T> Name for Pso<T> {
  fn name(&self) -> &String {
    &self.name
  }
}

impl<T: Clone> Data<T> for Pso<T> {
  fn data(&self) -> &Vec<(f64, Option<Vec<T>>)> {
    &self.data
  }

  fn add_data_impl(&mut self, datum: (f64, Option<Vec<T>>)) {
    self.data.push(datum);
  }
}

impl<T: Position + Velocity + Clone> DataExporter<T> for Pso<T> {
  fn out_directory(&self) -> &PathBuf {
    &self.out_directory
  }
}
