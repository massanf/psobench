extern crate nalgebra as na;
use crate::problem;
use crate::utils;
use nalgebra::DVector;
use problem::Problem;

pub trait ParticleTrait: Position + BestPosition + Velocity + Clone {
  fn new(problem: &mut Problem) -> Self
  where
    Self: Sized;
}

pub trait Position {
  fn init(&mut self, problem: &mut Problem) {
    self.set_pos(utils::random_init_pos(problem));
  }

  fn pos(&self) -> &DVector<f64>;
  fn set_pos(&mut self, pos: DVector<f64>);
}

pub trait BestPosition: Position {
  fn init(&mut self) {
    self.set_best_pos(self.pos().clone());
  }

  fn best_pos(&self) -> DVector<f64>;
  fn option_best_pos(&self) -> &Option<DVector<f64>>;
  fn set_best_pos(&mut self, pos: DVector<f64>);

  fn update_pos_and_best_pos(&mut self, pos: DVector<f64>, problem: &mut Problem) {
    self.set_pos(pos);
    self.update_best_pos(problem);
  }

  fn update_best_pos(&mut self, problem: &mut Problem) {
    // This function returns whether the personal best was updated.
    if self.option_best_pos().is_none() || problem.f(&self.pos()) < problem.f(&self.best_pos()) {
      self.set_best_pos(self.pos().clone());
    }
  }
}

pub trait Velocity: Position + BestPosition {
  fn init(&mut self, problem: &mut Problem) {
    self.set_vel(utils::random_init_vel(problem));
  }

  fn vel(&self) -> &DVector<f64>;
  fn set_vel(&mut self, vel: DVector<f64>);

  fn move_pos(&mut self, problem: &mut Problem) {
    let mut new_pos = self.pos().clone();
    let mut new_vel = self.vel().clone();

    // Check wall.
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
    self.set_pos(new_pos);
    self.update_best_pos(problem);
  }
}

pub trait Mass {
  fn set_mass(&mut self, mass: f64);
  fn mass(&self) -> f64;
}
