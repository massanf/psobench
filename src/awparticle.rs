extern crate nalgebra as na;

use crate::function;
use crate::particle;
use function::OptimizationProblem;
use nalgebra::DVector;
use particle::ParticleTrait;

use crate::rand::Rng;

pub struct AWParticle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  best_pos: Option<DVector<f64>>,
}

fn sigmoid(x: f64, m: f64) -> f64 {
  let a = 0.000035 * m;
  let b = 0.5;
  let c = 0.;
  let d = 1.5;
  b / (1.0 + (-a * (x - c)).exp()) + d
}

impl ParticleTrait for AWParticle {
  fn new(problem: &OptimizationProblem, dimensions: usize) -> AWParticle {
    let mut particle = AWParticle {
      pos: DVector::from_element(dimensions, 0.),
      vel: DVector::from_element(dimensions, 0.),
      best_pos: None,
    };
    particle.init(problem, dimensions);
    particle
  }

  fn pos(&self) -> &DVector<f64> {
    &self.pos
  }

  fn set_pos(&mut self, pos: DVector<f64>) {
    self.pos = pos;
  }

  fn best_pos(&self) -> DVector<f64> {
    self.best_pos.clone().unwrap()
  }

  fn option_best_pos(&self) -> &Option<DVector<f64>> {
    &self.best_pos
  }

  fn set_best_pos(&mut self, pos: DVector<f64>) {
    self.best_pos = Some(pos);
  }

  fn vel(&self) -> &DVector<f64> {
    &self.vel
  }

  fn set_vel(&mut self, vel: DVector<f64>) {
    self.vel = vel;
  }

  fn update_vel(&mut self, global_best_pos: &DVector<f64>) {
    let w = 0.8;
    let phi_p = sigmoid((self.best_pos() - self.pos()).norm(), 2.);
    let phi_g = sigmoid((global_best_pos - self.pos()).norm(), 2.);
    let mut rng = rand::thread_rng();
    let r_p: f64 = rng.gen_range(0.0..1.0);
    let r_g: f64 = rng.gen_range(0.0..1.0);
    self.set_vel(
      w * self.vel() + phi_p * r_p * (self.best_pos() - self.pos()) + phi_g * r_g * (global_best_pos - self.pos()),
    );
  }
}
