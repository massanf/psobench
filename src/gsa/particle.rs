// extern crate nalgebra as na;
// use crate::optimization_problem;
// use crate::particle_trait;
// use crate::ParamValue;
// use nalgebra::DVector;
// use optimization_problem::Problem;
// use particle_trait::ParticleTrait;
// use std::collections::HashMap;

// #[derive(Clone)]
// pub struct GSAParticle {
//   pos: DVector<f64>,
//   vel: DVector<f64>,
//   best_pos: Option<DVector<f64>>,
// }

// impl ParticleTrait for GSAParticle {
//   fn new(problem: &Problem) -> GSAParticle {
//     let mut particle = GSAParticle {
//       pos: DVector::from_element(problem.dim(), 0.),
//       vel: DVector::from_element(problem.dim(), 0.),
//       best_pos: None,
//     };
//     particle.init(problem);
//     particle
//   }

//   fn pos(&self) -> &DVector<f64> {
//     &self.pos
//   }

//   fn set_pos(&mut self, pos: DVector<f64>) {
//     self.pos = pos;
//   }

//   fn best_pos(&self) -> DVector<f64> {
//     self.best_pos.clone().unwrap()
//   }

//   fn option_best_pos(&self) -> &Option<DVector<f64>> {
//     &self.best_pos
//   }

//   fn set_best_pos(&mut self, pos: DVector<f64>) {
//     self.best_pos = Some(pos);
//   }

//   fn vel(&self) -> &DVector<f64> {
//     &self.vel
//   }

//   fn set_vel(&mut self, vel: DVector<f64>) {
//     self.vel = vel;
//   }

//   fn update_vel(&mut self, global_best_pos: &DVector<f64>, problem: &Problem, param: &HashMap<String, ParamValue>) {}
// }
