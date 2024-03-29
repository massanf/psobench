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
pub struct GSA<T: ParticleTrait> {
  name: String,
  problem: Problem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  global_worst_pos: Option<DVector<f64>>,
  m: Option<Vec<f64>>,
  g: f64,
  data: Vec<(f64, Vec<T>)>,
  out_directory: PathBuf,
  g0: f64,
  alpha: f64,
  epsilon: f64,
}

impl<T: ParticleTrait> PSOTrait<T> for GSA<T> {
  fn new(name: &str, problem: Problem, parameters: HashMap<String, ParamValue>, out_directory: PathBuf) -> GSA<T> {
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

    assert!(parameters.contains_key("g0"), "Key 'g0' not found.");
    let g0: f64;
    match parameters["g0"] {
      ParamValue::Float(val) => g0 = val,
      _ => {
        eprintln!("Error: parameter 'g0' should be of type Param::Float.");
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

    assert!(parameters.contains_key("epsilon"), "Key 'epsilon' not found.");
    let epsilon: f64;
    match parameters["epsilon"] {
      ParamValue::Float(val) => epsilon = val,
      _ => {
        eprintln!("Error: parameter 'epsilon' should be of type Param::Float.");
        std::process::exit(1);
      }
    }

    let mut gsa = GSA {
      name: name.to_owned(),
      problem,
      particles: Vec::new(),
      global_best_pos: None,
      global_worst_pos: None,
      m: None,
      g: 100.,
      data: Vec::new(),
      out_directory,
      g0,
      alpha,
      epsilon,
    };

    gsa.init(number_of_particles);
    gsa
  }

  fn init(&mut self, number_of_particles: usize) {
    let problem = &mut self.problem();
    let mut particles: Vec<T> = Vec::new();
    for _ in 0..number_of_particles {
      particles.push(ParticleTrait::new(problem));
    }

    let mut global_best_pos = None;
    let mut global_worst_pos = None;

    for particle in particles.clone() {
      if global_best_pos.is_none() || problem.f(&particle.pos()) < problem.f(global_best_pos.as_ref().unwrap()) {
        global_best_pos = Some(particle.pos().clone());
      }
      if global_worst_pos.is_none() || problem.f(&particle.pos()) > problem.f(global_worst_pos.as_ref().unwrap()) {
        global_worst_pos = Some(particle.pos().clone());
      }
    }

    self.particles = particles;
    self.global_best_pos = Some(global_best_pos.unwrap());
    self.global_worst_pos = Some(global_worst_pos.unwrap());
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

  fn calculate_vel(&mut self, idx: usize) -> DVector<f64> {
    assert!(idx < self.particles().len());

    let m = self.m.as_ref().unwrap().clone();
    let mut f: DVector<f64> = DVector::from_element(self.problem().dim(), 0.);
    for j in 0..self.particles().len() {
      if idx == j {
        continue;
      }
      let r = self.particles()[j].pos() - self.particles()[idx].pos();
      assert!(r.norm() + self.epsilon != 0.);
      f += self.g * (m[idx] * m[j]) / (r.norm() + self.epsilon) * r;
    }

    if m[idx] == 0. {
      return self.particles()[idx].vel().clone();
    }
    assert!(m[idx] != 0.);
    let a = f / m[idx];

    let mut rng = rand::thread_rng();
    let rand: f64 = rng.gen_range(0.0..1.0);

    let mut new_vel = rand * self.particles()[idx].vel() + a;

    // Check and account for V_{max} in each dimension.
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
    let global_best_pos = self.global_best_pos().clone();
    let gbest = self.problem().f(&global_best_pos);
    let particles = self.particles().clone();
    self.data.push((gbest, particles));
  }

  fn run(&mut self, iterations: usize) {
    for iter in 0..iterations {
      self.g = self.g0 * (-self.alpha * iter as f64 / iterations as f64).exp();

      // Calculate M
      let mut m_unscaled = Vec::new();
      let mut m_sum = 0.;
      let best = self.global_best_pos.as_ref().unwrap().clone();
      let worst = self.global_worst_pos.as_ref().unwrap().clone();
      assert!(self.problem().f(&best) != self.problem().f(&worst));
      for idx in 0..self.particles().len() {
        let p = self.particles()[idx].clone();
        let numerator = self.problem().f(p.pos()) - self.problem().f(&worst);
        let denominator = self.problem().f(&best) - self.problem().f(&worst);
        assert!(numerator <= 0.);
        assert!(denominator < 0.);
        let m_i = numerator / denominator;
        m_unscaled.push(m_i);
        m_sum += m_i;
      }
      let mut m = Vec::new();
      for idx in 0..self.particles().len() {
        m.push(m_unscaled[idx] / m_sum);
      }
      self.m = Some(m);

      // Calculate vels.
      let mut vels = Vec::new();
      for idx in 0..self.particles().len() {
        vels.push(self.calculate_vel(idx));
      }

      // Reset values for good measure.
      self.m = None;

      // Update the position, best and worst.
      let mut new_global_best_pos = self.global_best_pos.clone().unwrap();
      let mut new_global_worst_pos = self.global_worst_pos.clone().unwrap();
      for idx in 0..self.particles().len() {
        let problem = &mut self.problem().clone();
        self.particles_mut()[idx].set_vel(vels[idx].clone());
        let _ = self.particles_mut()[idx].update_pos(problem);
        let pos = self.particles()[idx].pos().clone();
        if self.problem().f(&pos) < self.problem().f(&new_global_best_pos) {
          new_global_best_pos = self.particles()[idx].pos().clone();
        }
        if self.problem().f(&pos) > self.problem().f(&new_global_worst_pos) {
          new_global_worst_pos = self.particles()[idx].pos().clone();
        }
      }
      self.global_best_pos = Some(new_global_best_pos);
      self.global_worst_pos = Some(new_global_worst_pos);

      // Save the data for current iteration.
      self.add_data();
    }
  }
}
