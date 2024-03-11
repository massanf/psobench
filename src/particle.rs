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
    // This function returns whether the personal best was updated.
    self.new_pos(self.pos() + self.vel(), problem)
  }

  fn best_pos(&self) -> DVector<f64>;
  fn option_best_pos(&self) -> &Option<DVector<f64>>;
  fn set_best_pos(&mut self, pos: DVector<f64>);

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

  fn eval(&mut self, problem: &OptimizationProblem) -> bool {
    // This function returns whether the personal best was updated.
    if self.option_best_pos().is_none() || problem.f(&self.pos()) < problem.f(&self.best_pos()) {
      self.set_best_pos(self.pos().clone());
      return true;
    }
    false
  }
}
