extern crate nalgebra as na;
use crate::particles::traits::{Behavior, BehaviorTrait, Edge, Mass, Position, Velocity};
use crate::problems;
use nalgebra::DVector;
use problems::Problem;

#[derive(Clone)]
pub struct FDOParticle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  mass: f64,
  behavior: Behavior,
}

impl FDOParticle {
  pub fn new(problem: &mut Problem, behavior: Behavior) -> FDOParticle {
    let mut particle = FDOParticle {
      pos: DVector::from_element(problem.dim(), 0.),
      vel: DVector::from_element(problem.dim(), 0.),
      mass: 0.,
      behavior,
    };
    Position::init(&mut particle, problem);
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

impl Velocity for FDOParticle {
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

impl Mass for FDOParticle {
  fn set_mass(&mut self, mass: f64) {
    self.mass = mass;
  }

  fn mass(&self) -> f64 {
    self.mass
  }
}

impl BehaviorTrait for FDOParticle {
  fn edge(&self) -> Edge {
    self.behavior.edge
  }

  fn vmax(&self) -> bool {
    self.behavior.vmax
  }
}
