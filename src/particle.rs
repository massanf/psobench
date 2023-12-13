extern crate nalgebra as na;

use nalgebra::DVector;
use std::fmt;

use crate::rand::Rng;
use crate::utils;

pub trait ParticleTrait {
  fn new(dimensions: usize) -> Self
  where
    Self: Sized;
  fn pos(&self) -> &DVector<f64>;
  fn pos_eval(&self) -> &f64;
  fn vel(&self) -> &DVector<f64>;
  fn best_pos(&self) -> &DVector<f64>;
  fn best_pos_eval(&self) -> &f64;
  fn update_vel(&mut self, global_best_pos: &DVector<f64>);
  fn update_pos(&mut self) -> bool;
  fn eval(&mut self) -> bool;
}

pub struct Particle {
  pos: DVector<f64>,
  pos_eval: f64,
  vel: DVector<f64>,
  best_pos: DVector<f64>,
  best_pos_eval: f64,
}

impl ParticleTrait for Particle {
  fn new(dimensions: usize) -> Particle {
    let b_lo: DVector<f64> = DVector::from_element(dimensions, -1.0);
    let b_up: DVector<f64> = DVector::from_element(dimensions, 1.0);

    let pos: DVector<f64> = utils::uniform_distribution(&b_lo, &b_up);
    let vel: DVector<f64> = utils::uniform_distribution(
      &DVector::from_iterator(dimensions, (&b_up - &b_lo).iter().map(|b| -b.abs())),
      &DVector::from_iterator(dimensions, (&b_up - &b_lo).iter().map(|b| b.abs())),
    );

    Particle {
      pos: pos.clone(),
      pos_eval: 0.,
      vel: vel,
      best_pos: pos,
      best_pos_eval: f64::MAX,
    }
  }

  fn pos(&self) -> &DVector<f64> {
    &self.pos
  }

  fn pos_eval(&self) -> &f64 {
    &self.pos_eval
  }

  fn vel(&self) -> &DVector<f64> {
    &self.vel
  }

  fn best_pos(&self) -> &DVector<f64> {
    &self.best_pos
  }

  fn best_pos_eval(&self) -> &f64 {
    &self.best_pos_eval
  }

  fn update_vel(&mut self, global_best_pos: &DVector<f64>) {
    let w = 0.8;
    let phi_p = 2.;
    let phi_g = 2.;
    let mut rng = rand::thread_rng();
    let r_p: f64 = rng.gen_range(0.0..1.0);
    let r_g: f64 = rng.gen_range(0.0..1.0);
    self.vel = w * &self.vel + phi_p * r_p * (&self.best_pos - &self.pos) + phi_g * r_g * (global_best_pos - &self.pos);
  }

  fn update_pos(&mut self) -> bool {
    // This function returns whether the personal best was updated.
    self.pos = &self.pos + &self.vel;
    self.eval()
  }

  fn eval(&mut self) -> bool {
    // This function returns whether the personal best was updated.
    self.pos_eval = self.pos.iter().map(|x| x.powi(2)).sum();
    if self.pos_eval < self.best_pos_eval {
      self.best_pos = self.pos.clone();
      self.best_pos_eval = self.pos_eval.clone();
      return true;
    }
    false
  }
}

// A util formatter that returns a string that is formatted to
// contain all the information about a particle including its
// `pos`, `pos_eval`, `vel`, `best_pos`, and `best_eval`.
impl fmt::Display for dyn ParticleTrait {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "{}",
      &format!(
        " pos:  [{}] ({:.3})\n vel:  [{}]\n best: [{}] ({:.3})\n",
        utils::format_dvector(self.pos()),
        self.pos_eval(),
        utils::format_dvector(&self.vel()),
        utils::format_dvector(&self.best_pos()),
        self.best_pos_eval(),
      )
    )
  }
}
