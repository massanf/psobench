use crate::particle_trait::{BestPosition, Position, Velocity};
use crate::problem;
use crate::pso_trait::{
  Data, DataExporter, GlobalBestPos, Name, OptimizationProblem, ParamValue, ParticleOptimizer, Particles,
};
use crate::rand::Rng;
use crate::utils;
use crate::PSOParticle;
use nalgebra::DVector;
use problem::Problem;
use std::collections::HashMap;
use std::mem;
use std::path::PathBuf;

#[derive(Clone)]
pub struct PSO<PSOParticle> {
  name: String,
  problem: Problem,
  particles: Vec<PSOParticle>,
  global_best_pos: Option<DVector<f64>>,
  data: Vec<(f64, Vec<PSOParticle>)>,
  out_directory: PathBuf,
  w: f64,
  phi_p: f64,
  phi_g: f64,
}

impl ParticleOptimizer<PSOParticle> for PSO<PSOParticle> {
  fn new(
    name: &str,
    problem: Problem,
    parameters: HashMap<String, ParamValue>,
    out_directory: PathBuf,
  ) -> PSO<PSOParticle> {
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

    assert!(parameters.contains_key("w"), "Key 'w' not found.");
    let w: f64;
    match parameters["w"] {
      ParamValue::Float(val) => w = val,
      _ => {
        eprintln!("Error");
        std::process::exit(1);
      }
    }

    assert!(parameters.contains_key("phi_p"), "Key 'phi_p' not found.");
    let phi_p: f64;
    match parameters["phi_p"] {
      ParamValue::Float(val) => phi_p = val,
      _ => {
        eprintln!("Error");
        std::process::exit(1);
      }
    }

    assert!(parameters.contains_key("phi_g"), "Key 'phi_g' not found.");
    let phi_g: f64;
    match parameters["phi_g"] {
      ParamValue::Float(val) => phi_g = val,
      _ => {
        eprintln!("Error");
        std::process::exit(1);
      }
    }

    let mut pso = PSO {
      name: name.to_owned(),
      problem,
      particles: Vec::new(),
      global_best_pos: None,
      data: Vec::new(),
      out_directory,
      w,
      phi_p,
      phi_g,
    };

    pso.init(number_of_particles);
    pso
  }

  fn init(&mut self, number_of_particles: usize) {
    let problem = &mut self.problem();
    let mut particles: Vec<PSOParticle> = Vec::new();
    for _ in 0..number_of_particles {
      particles.push(PSOParticle::new(problem));
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

      let mut new_global_best_pos = self.global_best_pos().clone();
      for idx in 0..self.particles().len() {
        let vel = self.calculate_vel(idx);
        let mut temp_problem = mem::replace(&mut self.problem, Problem::default());
        let particle = &mut self.particles_mut()[idx];
        particle.set_vel(vel);
        particle.move_pos(&mut temp_problem);
        particle.update_best_pos(&mut temp_problem);
        let best_pos = self.particles()[idx].best_pos().clone();
        if self.problem().f(&best_pos) < self.problem.f(&new_global_best_pos) {
          new_global_best_pos = self.particles()[idx].best_pos().clone();
        }
        self.problem = temp_problem;
      }
      self.global_best_pos = Some(new_global_best_pos);

      // Save the data for current iteration.
      self.add_data();
    }
  }
}

impl<T> Particles<T> for PSO<T> {
  fn particles(&self) -> &Vec<T> {
    &self.particles
  }

  fn particles_mut(&mut self) -> &mut Vec<T> {
    &mut self.particles
  }
}

impl<T> GlobalBestPos for PSO<T> {
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

impl<T> OptimizationProblem for PSO<T> {
  fn problem(&mut self) -> &mut Problem {
    &mut self.problem
  }
}

impl<T> Name for PSO<T> {
  fn name(&self) -> &String {
    &self.name
  }
}

impl<T: Clone> Data<T> for PSO<T> {
  fn data(&self) -> &Vec<(f64, Vec<T>)> {
    &self.data
  }

  fn add_data(&mut self) {
    let gbest = self.problem.f(&self.global_best_pos());
    let particles = self.particles.clone();
    self.data.push((gbest, particles));
  }
}

impl<T: Position + Velocity + Clone> DataExporter<T> for PSO<T> {
  fn out_directory(&self) -> &PathBuf {
    &self.out_directory
  }
}
