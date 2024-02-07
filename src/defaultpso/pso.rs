use crate::function;
use crate::particle::ParticleTrait;
use crate::pso::PSOTrait;
use function::OptimizationProblem;
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::DVector;

#[derive(Clone)]
pub struct DefaultPSO<'a, T: ParticleTrait> {
  name: String,
  problem: &'a OptimizationProblem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  data: Vec<(f64, Vec<T>)>,
}

impl<'a, T: ParticleTrait> PSOTrait<'a, T> for DefaultPSO<'a, T> {
  fn new(
    name: &str,
    problem: &'a OptimizationProblem,
    dimensions: usize,
    number_of_particles: usize,
  ) -> DefaultPSO<'a, T> {
    let mut particles: Vec<T> = Vec::new();

    for _ in 0..number_of_particles {
      particles.push(ParticleTrait::new(&problem, dimensions));
    }

    let mut pso = DefaultPSO {
      name: name.to_owned(),
      problem,
      particles,
      global_best_pos: None,
      data: Vec::new(),
    };

    pso.init(dimensions);
    pso
  }

  fn name(&self) -> &String {
    &self.name
  }

  fn particles(&self) -> &Vec<T> {
    &self.particles
  }

  fn init_particles(&mut self, problem: &OptimizationProblem, dimensions: usize) {
    for i in 0..self.particles.len() {
      self.particles[i].init(problem, dimensions);
    }
  }

  fn problem(&self) -> &OptimizationProblem {
    return &self.problem;
  }

  fn global_best_pos(&self) -> DVector<f64> {
    self.global_best_pos.clone().unwrap()
  }

  fn set_global_best_pos(&mut self, pos: DVector<f64>) {
    self.global_best_pos = Some(pos);
  }

  fn option_global_best_pos(&self) -> &Option<DVector<f64>> {
    &self.global_best_pos
  }

  fn data(&self) -> &Vec<(f64, Vec<T>)> {
    &self.data
  }

  fn init_data(&mut self) {
    self.data = Vec::new();
  }

  fn add_data(&mut self) {
    let gbest = self.problem().f(&self.global_best_pos());
    let particles = self.particles().clone();
    self.data.push((gbest, particles));
  }

  fn run(&mut self, iterations: usize) {
    // Initialize the progress bar.
    let progress = ProgressBar::new(iterations as u64);
    progress.set_style(
      ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {percent}% ({eta})")
        .progress_chars("=> "),
    );

    for _ in 0..iterations {
      let global_best_pos = self.global_best_pos();
      let mut new_global_best_pos = self.global_best_pos().clone();
      for particle in &mut self.particles {
        particle.update_vel(&global_best_pos);
        if particle.update_pos(&self.problem) {
          if self.problem.f(&particle.best_pos()) < self.problem.f(&new_global_best_pos) {
            new_global_best_pos = particle.best_pos().clone();
          }
        }
      }
      self.global_best_pos = Some(new_global_best_pos);

      // Save the data for current iteration.
      self.add_data();

      // Increment the progress bar.
      progress.inc(1);
    }

    // Finish the progress bar.
    progress.finish();
  }
}
