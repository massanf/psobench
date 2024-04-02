extern crate nalgebra as na;
use crate::particle_trait;
use crate::problem;
use crate::utils;
use nalgebra::DVector;
use particle_trait::ParticleTrait;
use problem::Problem;

#[derive(Clone)]
pub struct GSAParticle {
  pos: DVector<f64>,
  vel: DVector<f64>,
  best_pos: Option<DVector<f64>>,
}

impl ParticleTrait for GSAParticle {
  fn new(problem: &mut Problem) -> GSAParticle {
    let mut particle = GSAParticle {
      pos: DVector::from_element(problem.dim(), 0.),
      vel: DVector::from_element(problem.dim(), 0.),
      best_pos: None,
    };
    particle.init(problem);
    particle
  }

  fn init(&mut self, problem: &mut Problem) {
    let pos = utils::random_init_pos(problem);
    self.new_pos(pos.clone(), problem);
    self.set_best_pos(pos);
    let vel = DVector::from_element(problem.dim(), 0.);
    self.set_vel(vel);
  }

  fn pos(&self) -> &DVector<f64> {
    &self.pos
  }

  fn update_pos(&mut self, problem: &mut Problem) {
    self.new_pos(self.pos().clone() + self.vel().clone(), problem);
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
