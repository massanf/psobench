extern crate nalgebra as na;

use crate::particle;

use nalgebra::DVector;
use particle::ParticleTrait;

use crate::rand::Rng;
use crate::utils;

pub struct AWParticle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  best_pos: DVector<f64>,
}

fn sigmoid(x: f64, m: f64) -> f64 {
  let a = 0.000035 * m;
  let b = 0.5;
  let c = 0.;
  let d = 1.5;
  b / (1.0 + (-a * (x - c)).exp()) + d
}

impl ParticleTrait for AWParticle {
  fn new(f: &fn(&DVector<f64>) -> f64, dimensions: usize) -> AWParticle {
    let pos = utils::random_init_pos(dimensions);
    let mut particle = AWParticle {
      pos: pos.clone(),
      vel: utils::random_init_vel(dimensions),
      best_pos: pos,
    };

    particle.eval(f);
    particle
  }

  fn pos(&self) -> &DVector<f64> {
    &self.pos
  }

  fn set_pos(&mut self, pos: DVector<f64>) {
    self.pos = pos;
  }

  fn best_pos(&self) -> &DVector<f64> {
    &self.best_pos
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

  fn eval(&mut self, f: &fn(&DVector<f64>) -> f64) -> bool {
    // This function returns whether the personal best was updated.
    if f(&self.pos()) < f(&self.best_pos()) {
      self.best_pos = self.pos.clone();
      return true;
    }
    false
  }
}
