extern crate nalgebra as na;

use crate::particle;

use crate::function;
use function::OptimizationProblem;
use nalgebra::DVector;
use particle::ParticleTrait;

pub struct DefaultParticle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  best_pos: Option<DVector<f64>>,
}

impl ParticleTrait for DefaultParticle {
  fn new(problem: &OptimizationProblem, dimensions: usize) -> DefaultParticle {
    let mut particle = DefaultParticle {
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
}
