extern crate nalgebra as na;
use crate::particles::traits::{Behavior, BehaviorTrait, Edge, Mass, Particle, Position, Velocity};
use crate::problems;
use nalgebra::DVector;
use problems::Problem;

#[derive(Clone)]
pub struct GSAParticle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  mass: f64,
  behavior: Behavior,
}

impl Particle for GSAParticle {
  fn new(problem: &mut Problem, behavior: Behavior) -> GSAParticle {
    let mut particle = GSAParticle {
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

impl Position for GSAParticle {
  fn pos(&self) -> &DVector<f64> {
    &self.pos
  }

  fn set_pos(&mut self, pos: DVector<f64>) {
    self.pos = pos;
  }
}

impl Velocity for GSAParticle {
  fn init(&mut self, problem: &mut Problem) {
    self.set_vel(DVector::from_element(problem.dim(), 0.), problem);
  }

  fn vel(&self) -> &DVector<f64> {
    &self.vel
  }

  fn set_vel(&mut self, vel: DVector<f64>, problem: &mut Problem) {
    let mut new_vel = vel.clone();
    let v_max = (problem.domain().1 - problem.domain().0) / 2.;

    for e in new_vel.iter_mut() {
      if *e > v_max {
        *e = v_max;
      } else if *e < v_max {
        *e = -v_max;
      }
    }

    self.vel = vel;
  }

  fn move_pos(&mut self, problem: &mut Problem) {
    let mut new_pos = self.pos().clone();
    let mut new_vel = self.vel().clone();

    // Check wall and reflect.
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
    self.set_vel(new_vel, problem);

    // This function returns whether the personal best was updated.
    self.set_pos(new_pos);
  }
}

impl Mass for GSAParticle {
  fn set_mass(&mut self, mass: f64) {
    self.mass = mass;
  }

  fn mass(&self) -> f64 {
    self.mass
  }
}

impl BehaviorTrait for GSAParticle {
  fn edge(&self) -> Edge {
    self.behavior.edge
  }
}
