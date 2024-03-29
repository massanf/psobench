use crate::particle_trait::ParticleTrait;
use crate::problem;
use crate::pso_trait::PSOTrait;
use crate::pso_trait::ParamValue;
use crate::rand::Rng;
use crate::utils;
use nalgebra::DVector;
use problem::Problem;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone)]
pub struct PSO<T: ParticleTrait> {
  name: String,
  problem: Problem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  data: Vec<(f64, Vec<T>)>,
  out_directory: PathBuf,
  w: f64,
  phi_p: f64,
  phi_g: f64,
}

impl<T: ParticleTrait> PSOTrait<T> for PSO<T> {
  fn new(name: &str, problem: Problem, parameters: HashMap<String, ParamValue>, out_directory: PathBuf) -> PSO<T> {
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
    let mut particles: Vec<T> = Vec::new();
    for _ in 0..number_of_particles {
      particles.push(ParticleTrait::new(problem));
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

  fn name(&self) -> &String {
    &self.name
  }

  fn particles(&self) -> &Vec<T> {
    &self.particles
  }

  fn particles_mut(&mut self) -> &mut Vec<T> {
    &mut self.particles
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

  fn problem(&mut self) -> &mut Problem {
    &mut self.problem
  }

  fn out_directory(&self) -> &PathBuf {
    &self.out_directory
  }

  fn global_best_pos(&self) -> DVector<f64> {
    self.global_best_pos.clone().unwrap()
  }

  fn option_global_best_pos(&self) -> &Option<DVector<f64>> {
    &self.global_best_pos
  }

  fn data(&self) -> &Vec<(f64, Vec<T>)> {
    &self.data
  }

  fn add_data(&mut self) {
    let data = &self.global_best_pos().clone();
    let gbest = self.problem().f(data);
    let particles = self.particles().clone();
    self.data.push((gbest, particles));
  }

  fn run(&mut self, iterations: usize) {
    for _ in 0..iterations {
      let mut new_global_best_pos = self.global_best_pos().clone();
      for idx in 0..self.particles().len() {
        let vel = self.calculate_vel(idx);
        let problem = &mut self.problem().clone();
        self.particles_mut()[idx].set_vel(vel);
        if self.particles_mut()[idx].update_pos(problem) {
          if self.problem.f(&self.particles()[idx].best_pos()) < self.problem.f(&new_global_best_pos) {
            new_global_best_pos = self.particles()[idx].best_pos().clone();
          }
        }
      }
      self.global_best_pos = Some(new_global_best_pos);

      // Save the data for current iteration.
      self.add_data();
    }
  }
}
