extern crate nalgebra as na;
use crate::particles::traits::{Behavior, BehaviorTrait, Particle, Position, Velocity};
use crate::problems;
use nalgebra::DVector;
use problems::Problem;

#[derive(Clone)]
pub struct GaussianParticle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  behavior: Behavior,
}

impl Particle for GaussianParticle {
  fn new(problem: &mut Problem, behavior: Behavior) -> GaussianParticle {
    let mut particle = GaussianParticle {
      pos: DVector::from_element(problem.dim(), 0.),
      vel: DVector::from_element(problem.dim(), 0.),
      behavior,
    };
    Position::init(&mut particle, problem);
    Velocity::init(&mut particle, problem);
    particle
  }
}

impl Position for GaussianParticle {
  fn pos(&self) -> &DVector<f64> {
    &self.pos
  }

  fn set_pos(&mut self, pos: DVector<f64>) {
    self.pos = pos;
  }
}

impl Velocity for GaussianParticle {
  fn init(&mut self, problem: &mut Problem) {
    self.update_vel(DVector::from_element(problem.dim(), 0.), problem);
  }

  fn vel(&self) -> &DVector<f64> {
    &self.vel
  }

  fn set_vel(&mut self, vel: DVector<f64>) {
    self.vel = vel;
  }
}

impl BehaviorTrait for GaussianParticle {
  fn behavior(&self) -> Behavior {
    self.behavior
  }
}
