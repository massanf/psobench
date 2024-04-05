extern crate nalgebra as na;
use crate::particle_trait::{BestPosition, Position, Velocity};
use crate::problem;
use nalgebra::DVector;
use problem::Problem;

#[derive(Clone)]
pub struct PSOParticle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  best_pos: Option<DVector<f64>>,
}

impl PSOParticle {
  pub fn new(problem: &mut Problem) -> PSOParticle {
    let mut particle = PSOParticle {
      pos: DVector::from_element(problem.dim(), 0.),
      vel: DVector::from_element(problem.dim(), 0.),
      best_pos: None,
    };
    Position::init(&mut particle, problem);
    BestPosition::init(&mut particle);
    Velocity::init(&mut particle, problem);
    particle
  }
}

impl Position for PSOParticle {
  fn pos(&self) -> &DVector<f64> {
    &self.pos
  }

  fn set_pos(&mut self, pos: DVector<f64>) {
    self.pos = pos;
  }
}

impl BestPosition for PSOParticle {
  fn best_pos(&self) -> DVector<f64> {
    self.best_pos.clone().unwrap()
  }

  fn option_best_pos(&self) -> &Option<DVector<f64>> {
    &self.best_pos
  }

  fn set_best_pos(&mut self, pos: DVector<f64>) {
    self.best_pos = Some(pos);
  }
}

impl Velocity for PSOParticle {
  fn vel(&self) -> &DVector<f64> {
    &self.vel
  }

  fn set_vel(&mut self, vel: DVector<f64>) {
    self.vel = vel;
  }
}
