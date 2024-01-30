use crate::grapher;
use crate::particle;
use crate::utils;

use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::DVector;
use particle::ParticleTrait;
use std::fmt;

pub struct PSO<T: ParticleTrait> {
  f: fn(&DVector<f64>) -> f64,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  data: Vec<(f64, Vec<f64>)>,
}

impl<T: ParticleTrait> PSO<T> {
  pub fn new(f: fn(&DVector<f64>) -> f64, dimensions: usize, number_of_particles: usize) -> PSO<T> {
    let mut particles: Vec<T> = Vec::new();

    // Create and save particle of type T.
    for _ in 0..number_of_particles {
      particles.push(ParticleTrait::new(&f, dimensions));
    }

    let mut pso = PSO {
      f,
      particles,
      global_best_pos: None,
      data: Vec::new(),
    };
    pso.init(dimensions);
    pso
  }

  fn global_best_pos(&self) -> DVector<f64> {
    self.global_best_pos.clone().unwrap()
  }

  pub fn init(&mut self, dimensions: usize) {
    for particle in &mut self.particles {
      particle.init(&self.f, dimensions);
    }

    for particle in &self.particles {
      if self.global_best_pos.is_none() || (self.f)(&particle.pos()) < (self.f)(&self.global_best_pos()) {
        self.global_best_pos = Some(particle.pos().clone());
      }
    }
  }

  pub fn run(&mut self, iterations: usize) -> f64 {
    // Initialize the progress bar.
    let progress = ProgressBar::new(iterations as u64);
    progress.set_style(
      ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {percent}% ({eta})")
        .progress_chars("=> "),
    );

    // Initialize data storing variable.

    for _ in 0..iterations {
      // Variable to store data for this iteration.
      let mut iteration_data: Vec<f64> = Vec::new();

      let global_best_pos = self.global_best_pos();
      let mut new_global_best_pos = self.global_best_pos().clone();
      for particle in &mut self.particles {
        particle.update_vel(&global_best_pos);
        if particle.update_pos(&self.f) {
          if (self.f)(&particle.best_pos()) < (self.f)(&new_global_best_pos) {
            new_global_best_pos = particle.best_pos().clone();
          }
        }
        iteration_data.push((self.f)(&particle.pos()).clone());
      }
      self.global_best_pos = Some(new_global_best_pos);

      // Save the data for current iteration.
      self.data.push(((self.f)(&self.global_best_pos()), iteration_data));

      // Increment the progress bar.
      progress.inc(1);
    }

    // Finish the progress bar.
    progress.finish();

    // Draw and save the graph for the run.
    let _ = grapher::create_progress_graph(&self.data);

    (self.f)(&self.global_best_pos())
  }
}

// A util formatter that returns a formatted string that prints out
// current information about all of the particles.
impl<T: ParticleTrait + std::fmt::Display> fmt::Display for PSO<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut result = String::new();
    for (i, particle) in self.particles.iter().enumerate() {
      result.push_str(&format!("Particle {}:\n {}", i, particle));
    }
    result.push_str(&format!(
      "global best pos: [{}] ({:.3})",
      utils::format_dvector(&self.global_best_pos()),
      (self.f)(&self.global_best_pos()),
      // self.global_best_pos_eval,
    ));
    write!(f, "{}", result)
  }
}
