use crate::particle_trait::ParticleTrait;
use crate::problem;
use crate::pso_trait::PSOTrait;
use crate::pso_trait::ParamValue;
use crate::utils;
use nalgebra::DVector;
use problem::Problem;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone)]
pub struct CFO<T: ParticleTrait> {
  name: String,
  problem: Problem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  data: Vec<(f64, Vec<T>)>,
  out_directory: PathBuf,
  g: f64,
  alpha: f64,
  beta: f64,
}

impl<T: ParticleTrait> PSOTrait<T> for CFO<T> {
  fn new(name: &str, problem: Problem, parameters: HashMap<String, ParamValue>, out_directory: PathBuf) -> CFO<T> {
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

    assert!(parameters.contains_key("g"), "Key 'alpha' not found.");
    let g: f64;
    match parameters["g"] {
      ParamValue::Float(val) => g = val,
      _ => {
        eprintln!("Error: parameter 'g' should be of type Param::Float.");
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

    assert!(parameters.contains_key("beta"), "Key 'alpha' not found.");
    let beta: f64;
    match parameters["beta"] {
      ParamValue::Float(val) => beta = val,
      _ => {
        eprintln!("Error: parameter 'beta' should be of type Param::Float.");
        std::process::exit(1);
      }
    }

    let mut cfo = CFO {
      name: name.to_owned(),
      problem,
      particles: Vec::new(),
      global_best_pos: None,
      data: Vec::new(),
      out_directory,
      g,
      alpha,
      beta,
    };

    cfo.init(number_of_particles);
    cfo
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

    utils::create_directory(self.out_directory().to_path_buf(), true, false);
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

  fn calculate_vel(&mut self, i: usize) -> DVector<f64> {
    assert!(i < self.particles().len());

    let mut a: DVector<f64> = DVector::from_element(self.problem().dim(), 0.);
    let pos_i = self.particles()[i].pos().clone();
    let m_i = self.problem().f(&pos_i);

    for j in 0..self.particles().len() {
      let pos_j = self.particles()[j].pos().clone();
      let m_j = self.problem().f(&pos_j);
      if i == j {
        continue;
      }

      if m_j - m_i > 0. {
        let r = self.particles()[j].pos() - self.particles()[i].pos();
        a += self.g * (m_j - m_i).powf(self.alpha) * r.clone() / (r.norm().powf(self.beta) + std::f64::EPSILON);
      }
    }

    let mut new_vel = self.particles()[i].vel() + 0.5 * a;

    for e in new_vel.iter_mut() {
      if *e > self.problem().domain().1 - self.problem().domain().0 {
        *e = self.problem().domain().1 - self.problem().domain().0 / 2.;
      } else if *e < self.problem().domain().0 - self.problem().domain().1 {
        *e = self.problem().domain().0 - self.problem().domain().1 / 2.;
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
    let global_best_pos = self.global_best_pos().clone();
    let gbest = self.problem().f(&global_best_pos);
    let particles = self.particles().clone();
    self.data.push((gbest, particles));
  }

  fn run(&mut self, iterations: usize) {
    for _ in 0..iterations {
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
        let problem = &mut self.problem().clone();
        let particle = &mut self.particles_mut()[idx];
        particle.set_vel(vels[idx].clone());
        let _ = particle.update_pos(problem);
        let pos = particle.pos().clone();
        if self.problem().f(&pos) < self.problem().f(&new_global_best_pos) {
          new_global_best_pos = self.particles()[idx].pos().clone();
        }
      }
      self.global_best_pos = Some(new_global_best_pos);

      // Save the data for current iteration.
      self.add_data();
    }
  }
}
