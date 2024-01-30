use crate::particle;
use crate::utils;

use crate::function;
use function::OptimizationProblem;
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::DVector;
use particle::ParticleTrait;
use std::fmt;

pub struct PSO<'a, T: ParticleTrait> {
  problem: &'a OptimizationProblem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  data: Vec<f64>,
}

impl<T: ParticleTrait> PSO<'_, T> {
  pub fn new(problem: &OptimizationProblem, dimensions: usize, number_of_particles: usize) -> PSO<T> {
    let mut particles: Vec<T> = Vec::new();

    for _ in 0..number_of_particles {
      particles.push(ParticleTrait::new(&problem, dimensions));
    }

    let mut pso = PSO {
      problem,
      particles,
      global_best_pos: None,
      data: Vec::new(),
    };

    pso.init(dimensions);
    pso
  }

  pub fn global_best_pos(&self) -> DVector<f64> {
    self.global_best_pos.clone().unwrap()
  }

  pub fn data(&self) -> &Vec<f64> {
    &self.data
  }

  pub fn init(&mut self, dimensions: usize) {
    for particle in &mut self.particles {
      particle.init(&self.problem, dimensions);
    }

    for particle in &self.particles {
      if self.global_best_pos.is_none() || self.problem.f(&particle.pos()) < self.problem.f(&self.global_best_pos()) {
        self.global_best_pos = Some(particle.pos().clone());
      }
    }
  }

  pub fn run(&mut self, iterations: usize) {
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
      self.data.push(self.problem.f(&self.global_best_pos()));

      // Increment the progress bar.
      progress.inc(1);
    }

    // Finish the progress bar.
    progress.finish();
  }
}

// A util formatter that returns a formatted string that prints out
// current information about all of the particles.
impl<T: ParticleTrait + std::fmt::Display> fmt::Display for PSO<'_, T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut result = String::new();
    for (i, particle) in self.particles.iter().enumerate() {
      result.push_str(&format!("Particle {}:\n {}", i, particle));
    }
    result.push_str(&format!(
      "global best pos: [{}] ({:.3})",
      utils::format_dvector(&self.global_best_pos()),
      self.problem.f(&self.global_best_pos()),
      // self.global_best_pos_eval,
    ));
    write!(f, "{}", result)
  }
}
