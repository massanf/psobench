extern crate nalgebra as na;
use crate::optimization_problem;
use crate::pso_trait::Param;
use crate::rand::Rng;
use crate::utils;
use nalgebra::DVector;
use optimization_problem::Problem;
use std::collections::HashMap;

pub trait ParticleTrait: Clone {
  fn new(problem: &Problem) -> Self
  where
    Self: Sized;

  fn init(&mut self, problem: &Problem) {
    let pos = utils::random_init_pos(problem);
    self.new_pos(pos.clone(), problem);
    self.set_best_pos(pos);
    self.set_vel(utils::random_init_vel(problem));
  }

  fn pos(&self) -> &DVector<f64>;
  fn set_pos(&mut self, pos: DVector<f64>);
  fn new_pos(&mut self, pos: DVector<f64>, problem: &Problem) -> bool {
    self.set_pos(pos);
    self.eval(problem)
  }

  fn update_pos(&mut self, problem: &Problem) -> bool {
    let mut new_pos = self.pos().clone();
    let mut new_vel = self.vel().clone();
    for (i, e) in new_pos.iter_mut().enumerate() {
      if self.pos()[i] + self.vel()[i] < problem.domain().0 {
        *e = 2. * problem.domain().0 - self.vel()[i] - self.pos()[i];
        new_vel[i] = -new_vel[i];
      } else if self.pos()[i] + self.vel()[i] > problem.domain().1 {
        *e = 2. * problem.domain().1 - self.vel()[i] - self.pos()[i];
        new_vel[i] = -new_vel[i];
      } else {
        *e = self.pos()[i] + self.vel()[i];
      }
    }

    // Set new velocity, as it may have hit a wall
    self.set_vel(new_vel);

    // This function returns whether the personal best was updated.
    self.new_pos(new_pos, problem)
  }

  fn best_pos(&self) -> DVector<f64>;
  fn option_best_pos(&self) -> &Option<DVector<f64>>;
  fn set_best_pos(&mut self, pos: DVector<f64>);

  fn vel(&self) -> &DVector<f64>;
  fn set_vel(&mut self, vel: DVector<f64>);

  fn update_vel(&mut self, global_best_pos: &DVector<f64>, problem: &Problem, param: &HashMap<String, Param>) {
    let mut rng = rand::thread_rng();
    let r_p: f64 = rng.gen_range(0.0..1.0);
    let r_g: f64 = rng.gen_range(0.0..1.0);

    assert!(param.contains_key("w"), "Key 'w' not found.");
    let w: f64;
    match param["w"] {
      Param::Numeric(val) => w = val,
      _ => {
        eprintln!("Error");
        std::process::exit(1);
      }
    }

    assert!(param.contains_key("phi_p"), "Key 'phi_p' not found.");
    let phi_p: f64;
    match param["phi_p"] {
      Param::Numeric(val) => phi_p = val,
      _ => {
        eprintln!("Error");
        std::process::exit(1);
      }
    }

    assert!(param.contains_key("phi_g"), "Key 'phi_g' not found.");
    let phi_g: f64;
    match param["phi_g"] {
      Param::Numeric(val) => phi_g = val,
      _ => {
        eprintln!("Error");
        std::process::exit(1);
      }
    }

    let mut new_vel =
      w * self.vel() + phi_p * r_p * (self.best_pos() - self.pos()) + phi_g * r_g * (global_best_pos - self.pos());
    for e in new_vel.iter_mut() {
      if *e > problem.domain().1 - problem.domain().0 {
        *e = problem.domain().1 - problem.domain().0;
      } else if *e < problem.domain().0 - problem.domain().1 {
        *e = problem.domain().0 - problem.domain().1;
      }
    }
    self.set_vel(new_vel);
  }

  fn eval(&mut self, problem: &Problem) -> bool {
    // This function returns whether the personal best was updated.
    if self.option_best_pos().is_none() || problem.f(&self.pos()) < problem.f(&self.best_pos()) {
      self.set_best_pos(self.pos().clone());
      return true;
    }
    false
  }
}
