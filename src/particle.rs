extern crate nalgebra as na;

use nalgebra::DVector;
use std::fmt;

use crate::rand::Rng;
use crate::utils;

pub trait ParticleTrait {
  fn new(f: &fn(&DVector<f64>) -> f64, dimensions: usize) -> Self
  where
    Self: Sized;

  fn pos(&self) -> &DVector<f64>;
  fn set_pos(&mut self, pos: DVector<f64>);
  fn update_pos(&mut self, f: &fn(&DVector<f64>) -> f64) -> bool {
    // This function returns whether the personal best was updated.
    self.set_pos(self.pos() + self.vel());
    self.eval(f)
  }

  fn best_pos(&self) -> &DVector<f64>;

  fn vel(&self) -> &DVector<f64>;
  fn set_vel(&mut self, vel: DVector<f64>);
  fn update_vel(&mut self, global_best_pos: &DVector<f64>) {
    let w = 0.8;
    let phi_p = 2.;
    let phi_g = 2.;
    let mut rng = rand::thread_rng();
    let r_p: f64 = rng.gen_range(0.0..1.0);
    let r_g: f64 = rng.gen_range(0.0..1.0);
    self.set_vel(
      w * self.vel() + phi_p * r_p * (self.best_pos() - self.pos()) + phi_g * r_g * (global_best_pos - self.pos()),
    );
  }

  fn eval(&mut self, f: &fn(&DVector<f64>) -> f64) -> bool;
}

pub struct Particle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  best_pos: DVector<f64>,
}

impl ParticleTrait for Particle {
  fn new(f: &fn(&DVector<f64>) -> f64, dimensions: usize) -> Particle {
    let b_lo: DVector<f64> = DVector::from_element(dimensions, -1.0);
    let b_up: DVector<f64> = DVector::from_element(dimensions, 1.0);

    let pos: DVector<f64> = utils::uniform_distribution(&b_lo, &b_up);
    let vel: DVector<f64> = utils::uniform_distribution(
      &DVector::from_iterator(dimensions, (&b_up - &b_lo).iter().map(|b| -b.abs())),
      &DVector::from_iterator(dimensions, (&b_up - &b_lo).iter().map(|b| b.abs())),
    );

    let mut particle = Particle {
      pos: pos.clone(),
      vel: vel.clone(),
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

  fn vel(&self) -> &DVector<f64> {
    &self.vel
  }

  fn set_vel(&mut self, vel: DVector<f64>) {
    self.vel = vel;
  }

  fn best_pos(&self) -> &DVector<f64> {
    &self.best_pos
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

// A util formatter that returns a string that is formatted to
// contain all the information about a particle including its
// `pos`, `pos_eval`, `vel`, `best_pos`, and `best_eval`.
impl fmt::Display for dyn ParticleTrait {
  fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      fmt,
      "{}",
      &format!(
        " pos:  [{}] \n vel:  [{}]\n best: [{}] \n",
        utils::format_dvector(self.pos()),
        utils::format_dvector(&self.vel()),
        utils::format_dvector(&self.best_pos()),
      )
    )
  }
}
