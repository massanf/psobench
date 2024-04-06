extern crate nalgebra as na;
use crate::particle_trait::Mass;
use crate::particle_trait::{BestPosition, Position, Velocity};
use crate::problem;
use nalgebra::DVector;
use problem::Problem;

#[derive(Clone)]
pub struct FDOParticle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  mass: f64,
  best_pos: Option<DVector<f64>>,
}

impl FDOParticle {
  pub fn new(problem: &mut Problem) -> FDOParticle {
    let mut particle = FDOParticle {
      pos: DVector::from_element(problem.dim(), 0.),
      vel: DVector::from_element(problem.dim(), 0.),
      mass: 0.,
      best_pos: None,
    };
    Position::init(&mut particle, problem);
    BestPosition::init(&mut particle);
    Velocity::init(&mut particle, problem);
    particle
  }
}

impl Position for FDOParticle {
  fn pos(&self) -> &DVector<f64> {
    &self.pos
  }

  fn set_pos(&mut self, pos: DVector<f64>) {
    self.pos = pos;
  }
}

impl BestPosition for FDOParticle {
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

impl Velocity for FDOParticle {
  fn init(&mut self, problem: &mut Problem) {
    self.set_vel(DVector::from_element(problem.dim(), 0.));
  }

  fn vel(&self) -> &DVector<f64> {
    &self.vel
  }

  fn set_vel(&mut self, vel: DVector<f64>) {
    self.vel = vel;
  }

  fn move_pos(&mut self, problem: &mut Problem) {
    self.set_pos(self.pos().clone() + self.vel().clone());
    self.update_best_pos(problem);
  }
}

impl Mass for FDOParticle {
  fn set_mass(&mut self, mass: f64) {
    self.mass = mass;
  }

  fn mass(&self) -> f64 {
    self.mass
  }
}
