use crate::function;
use crate::particle;
use csv::Writer;
use function::OptimizationProblem;
use nalgebra::DVector;
use particle::ParticleTrait;
use std::error::Error;
use std::fs::File;
use std::path::Path;

pub trait PSOTrait<'a, T: ParticleTrait> {
  fn new(problem: &'a OptimizationProblem, dimensions: usize, number_of_particles: usize) -> Self
  where
    Self: Sized;

  fn init(&mut self, dimensions: usize) {
    let problem = self.problem().clone();
    let mut global_best_pos = None;
    for particle in &mut self.particles().iter_mut() {
      particle.init(&problem, dimensions);
    }
    for particle in self.particles() {
      if global_best_pos.is_none() || problem.f(&particle.pos()) < problem.f(global_best_pos.as_ref().unwrap()) {
        global_best_pos = Some(particle.pos().clone());
      }
    }
    self.set_global_best_pos(global_best_pos.unwrap());
    self.init_data();
    self.add_data(self.problem().f(&self.global_best_pos()));
  }

  fn particles(&mut self) -> &mut Vec<T>;

  fn problem(&self) -> &OptimizationProblem;

  fn global_best_pos(&self) -> DVector<f64>;
  fn set_global_best_pos(&mut self, pos: DVector<f64>);
  fn option_global_best_pos(&self) -> &Option<DVector<f64>>;

  fn data(&self) -> &Vec<f64>;
  fn init_data(&mut self);
  fn add_data(&mut self, datum: f64);

  fn run(&mut self, iterations: usize);

  fn export_global_best_progress(&self, file_path: &Path) -> Result<(), Box<dyn Error>> {
    let file = File::create(file_path)?;
    let mut wtr = Writer::from_writer(file);

    wtr.serialize(("global_best_pos",))?;
    for value in self.data() {
      wtr.serialize((value,))?;
    }

    wtr.flush()?;
    Ok(())
  }
}
