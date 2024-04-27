extern crate nalgebra as na;
use crate::problems;
use crate::utils;
use nalgebra::DVector;
use problems::Problem;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Behavior {
  pub edge: Edge,
  pub vmax: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub enum Edge {
  Reflect,
  Pass,
  Cycle,
}

pub trait Particle {
  fn new(problem: &mut Problem, behavior: Behavior) -> Self;
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

  fn update_best_pos(&mut self, problem: &mut Problem) {
    // This function returns whether the personal best was updated.
    if self.option_best_pos().is_none() || problem.f(self.pos()) < problem.f(&self.best_pos()) {
      self.set_best_pos(self.pos().clone());
    }
  }
}

pub trait Velocity: Position + BehaviorTrait {
  fn init(&mut self, problem: &mut Problem) {
    self.update_vel(utils::random_init_vel(problem), problem);
  }

  fn vel(&self) -> &DVector<f64>;
  fn set_vel(&mut self, vel: DVector<f64>);
  fn update_vel(&mut self, vel: DVector<f64>, problem: &mut Problem) {
    let vmax = problem.domain().1 - problem.domain().0;
    if self.behavior().vmax && vel.norm() > vmax {
      self.set_vel(vel.clone() / vel.norm() * vmax);
    } else {
      self.set_vel(vel);
    }
  }

  fn move_pos(&mut self, problem: &mut Problem) {
    match self.behavior().edge {
      Edge::Reflect => {
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
        self.set_pos(new_pos);

        // Set new velocity, as it may have hit a wall
        self.update_vel(new_vel, problem);
      }
      Edge::Pass => {
        self.set_pos(self.pos().clone() + self.vel().clone());
      }
      Edge::Cycle => {
        let mut new_pos = self.pos().clone() + self.vel().clone();

        // Check wall.
        let width = problem.domain().1 - problem.domain().0;
        for e in new_pos.iter_mut() {
          while *e < problem.domain().0 {
            *e += width;
          }
          while *e > problem.domain().1 {
            *e -= width;
          }
        }

        self.set_pos(new_pos);
      }
    }
  }
}

pub trait Mass: BehaviorTrait {
  fn mass(&self) -> f64;
  fn set_mass(&mut self, mass: f64);
}

pub trait BehaviorTrait {
  fn behavior(&self) -> Behavior;
}
