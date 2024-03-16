use crate::function;
use crate::particle_trait;
use function::OptimizationProblem;
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::DVector;
use particle_trait::ParticleTrait;
use serde_json::json;
use std::fs;
use std::path::Path;

pub trait PSOTrait<'a, T: ParticleTrait> {
  fn new(name: &str, problem: &'a OptimizationProblem, dimensions: usize, number_of_particles: usize) -> Self
  where
    Self: Sized;

  fn init(&mut self) {
    let problem = self.problem().clone();
    let mut global_best_pos = None;
    self.init_particles(&problem, self.dimensions().clone());
    for particle in self.particles() {
      if global_best_pos.is_none() || problem.f(&particle.pos()) < problem.f(global_best_pos.as_ref().unwrap()) {
        global_best_pos = Some(particle.pos().clone());
      }
    }
    self.set_global_best_pos(global_best_pos.unwrap());
    self.add_data();
  }

  fn name(&self) -> &String;

  fn particles(&self) -> &Vec<T>;
  fn init_particles(&mut self, problem: &OptimizationProblem, dimensions: usize);

  fn dimensions(&self) -> &usize;
  fn problem(&self) -> &OptimizationProblem;

  fn global_best_pos(&self) -> DVector<f64>;
  fn set_global_best_pos(&mut self, pos: DVector<f64>);
  fn option_global_best_pos(&self) -> &Option<DVector<f64>>;

  fn data(&self) -> &Vec<(f64, Vec<T>)>;
  fn add_data(&mut self);

  fn run(&mut self, iterations: usize);
  fn experiment(&mut self, trials: usize, iterations: usize) {
    // Initialize the progress bar.
    let progress = ProgressBar::new(trials as u64);
    progress.set_style(
      ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {percent}% ({eta})")
        .progress_chars("=> "),
    );

    for _ in 0..trials {
      self.init();
      self.run(iterations);

      // Increment the progress bar.
      progress.inc(1);
    }
  }

  fn save_history(&self, file_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the progress bar.
    let progress = ProgressBar::new(self.data().len() as u64);
    progress.set_style(
      ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {percent}% ({eta})")
        .progress_chars("=> "),
    );

    // Serialize it to a JSON string
    let mut vec_data = Vec::new();
    for t in 0..self.data().len() {
      let mut iter_data = Vec::new();
      for i in 0..self.data()[t].1.len() {
        let pos = self.data()[t].1[i].pos().clone();
        iter_data.push(json!({
          "fitness": self.problem().f(&pos),
          "vel": self.data()[t].1[i].vel().as_slice(),
          "pos": self.data()[t].1[i].pos().as_slice(),
        }));
      }
      vec_data.push(json!({
        "global_best_fitness": self.data()[t].0,
        "particles": iter_data
      }));
    }
    progress.inc(1);

    let serialized = serde_json::to_string(&json!(vec_data))?;

    fs::write(file_path, serialized)?;
    Ok(())
  }
}
