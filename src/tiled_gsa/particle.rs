extern crate nalgebra as na;
use crate::particle_trait::Mass;
use crate::particle_trait::{Position, Velocity};
use crate::problem;
use nalgebra::DVector;
use problem::Problem;

#[derive(Clone)]
pub struct TiledGSAParticle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  mass: f64,
}

impl TiledGSAParticle {
  pub fn new(problem: &mut Problem) -> TiledGSAParticle {
    let mut particle = TiledGSAParticle {
      pos: DVector::from_element(problem.dim(), 0.),
      vel: DVector::from_element(problem.dim(), 0.),
      mass: 0.,
    };
    Position::init(&mut particle, problem);
    Velocity::init(&mut particle, problem);
    particle
  }
}

impl Position for TiledGSAParticle {
  fn pos(&self) -> &DVector<f64> {
    &self.pos
  }

  fn set_pos(&mut self, pos: DVector<f64>) {
    self.pos = pos;
  }
}

impl Velocity for TiledGSAParticle {
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
    let mut new_pos = self.pos().clone();

    // Check wall.
    for (i, e) in new_pos.iter_mut().enumerate() {
      *e = self.pos()[i] + self.vel()[i];
      if *e < problem.domain().0 {
        *e += problem.domain().1 - problem.domain().0;
      }
      if *e > problem.domain().1 {
        *e -= problem.domain().1 - problem.domain().0;
      }
    }

    // This function returns whether the personal best was updated.
    self.set_pos(new_pos);
  }
}

impl Mass for TiledGSAParticle {
  fn set_mass(&mut self, mass: f64) {
    self.mass = mass;
  }

  fn mass(&self) -> f64 {
    self.mass
  }
}
