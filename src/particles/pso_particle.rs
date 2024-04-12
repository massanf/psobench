extern crate nalgebra as na;
use crate::particles::traits::{Behavior, BehaviorTrait, BestPosition, Edge, Particle, Position, Velocity};
use crate::problems;
use nalgebra::DVector;
use problems::Problem;

#[derive(Clone)]
pub struct PsoParticle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  best_pos: Option<DVector<f64>>,
  behavior: Behavior,
}

impl Particle for PsoParticle {
  fn new(problem: &mut Problem, behavior: Behavior) -> PsoParticle {
    let mut particle = PsoParticle {
      pos: DVector::from_element(problem.dim(), 0.),
      vel: DVector::from_element(problem.dim(), 0.),
      best_pos: None,
      behavior,
    };
    Position::init(&mut particle, problem);
    BestPosition::init(&mut particle);
    Velocity::init(&mut particle, problem);
    particle
  }
}

impl Position for PsoParticle {
  fn pos(&self) -> &DVector<f64> {
    &self.pos
  }

  fn set_pos(&mut self, pos: DVector<f64>) {
    self.pos = pos;
  }
}

impl BestPosition for PsoParticle {
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

impl Velocity for PsoParticle {
  fn vel(&self) -> &DVector<f64> {
    &self.vel
  }

  fn set_vel(&mut self, vel: DVector<f64>) {
    self.vel = vel;
  }
}

impl BehaviorTrait for PsoParticle {
  fn edge(&self) -> Edge {
    self.behavior.edge
  }

  fn vmax(&self) -> bool {
    self.behavior.vmax
  }
}
