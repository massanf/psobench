extern crate nalgebra as na;
use crate::particles::traits::{Behavior, BehaviorTrait, Edge, Mass, Particle, Position, Velocity};
use crate::problems;
use nalgebra::DVector;
use problems::Problem;
use std::f64::consts::E;

#[derive(Clone)]
pub struct IgsaParticle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  mass: f64,
  behavior: Behavior,
}

impl Particle for IgsaParticle {
  fn new(problem: &mut Problem, behavior: Behavior) -> IgsaParticle {
    let mut particle = IgsaParticle {
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

impl Position for IgsaParticle {
  fn pos(&self) -> &DVector<f64> {
    &self.pos
  }

  fn set_pos(&mut self, pos: DVector<f64>) {
    self.pos = pos;
  }
}

impl Velocity for IgsaParticle {
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

impl Mass for IgsaParticle {
  fn set_mass(&mut self, mass: f64) {
    let scale = 1.;
    self.mass = 1.0 / (1.0 + E.powf(-(scale * mass - scale / 2.)));
  }

  fn mass(&self) -> f64 {
    self.mass
  }
}

impl BehaviorTrait for IgsaParticle {
  fn edge(&self) -> Edge {
    self.behavior.edge
  }

  fn vmax(&self) -> bool {
    self.behavior.vmax
  }
}
