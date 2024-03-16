extern crate nalgebra as na;

use crate::function;
use function::OptimizationProblem;
use nalgebra::DVector;

use crate::rand::Rng;
use crate::utils;

pub trait ParticleTrait: Clone {
  fn new(problem: &OptimizationProblem, dimensions: usize) -> Self
  where
    Self: Sized;

  fn init(&mut self, problem: &OptimizationProblem, dimensions: usize) {
    let pos = utils::random_init_pos(dimensions, problem);
    self.new_pos(pos.clone(), problem);
    self.set_best_pos(pos);
    self.set_vel(utils::random_init_vel(dimensions, problem));
  }

  fn pos(&self) -> &DVector<f64>;
  fn set_pos(&mut self, pos: DVector<f64>);
  fn new_pos(&mut self, pos: DVector<f64>, problem: &OptimizationProblem) -> bool {
    self.set_pos(pos);
    self.eval(problem)
  }

  fn update_pos(&mut self, problem: &OptimizationProblem) -> bool {
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

  fn update_vel(&mut self, global_best_pos: &DVector<f64>, problem: &OptimizationProblem) {
    let w = 0.8;
    let phi_p = 1.;
    let phi_g = 2.;
    let mut rng = rand::thread_rng();
    let r_p: f64 = rng.gen_range(0.0..1.0);
    let r_g: f64 = rng.gen_range(0.0..1.0);

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

  fn eval(&mut self, problem: &OptimizationProblem) -> bool {
    // This function returns whether the personal best was updated.
    if self.option_best_pos().is_none() || problem.f(&self.pos()) < problem.f(&self.best_pos()) {
      self.set_best_pos(self.pos().clone());
      return true;
    }
    false
  }
}
