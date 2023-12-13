use crate::grapher;
use crate::particle;
use crate::utils;

use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::DVector;
use particle::ParticleTrait;
use std::fmt;

pub struct PSO<T: ParticleTrait> {
  particles: Vec<T>,
  global_best_pos: DVector<f64>,
  global_best_pos_eval: f64,
}

impl<T: ParticleTrait> PSO<T> {
  pub fn new(dimensions: usize, number_of_particles: usize) -> PSO<T> {
    // When calling `new`, make sure to call `init()` right after to
    // properly initialize all of the variables.
    // TODO: Fix this.
    let mut particles = Vec::new();

    // Create and save particle of type T.
    for _ in 0..number_of_particles {
      let mut particle: T = ParticleTrait::new(dimensions);

      // `new` does not automatically call `eval()`,
      // so we have to make sure to call it here.
      // TODO: Fix this.
      particle.eval();

      particles.push(particle);
    }

    PSO {
      particles,
      // This is initialized with `-1`, which is why the `init` has to be
      // called after `new`.
      // TODO: Fix this.
      global_best_pos: DVector::from_element(dimensions, -1.),
      global_best_pos_eval: f64::MAX,
    }
  }

  pub fn init(&mut self) {
    for particle in &mut self.particles {
      if particle.pos_eval() < &self.global_best_pos_eval {
        // `pos_eval()` returns `true` if its personal best was updated,
        // in which case the global best should be examined as well.
        self.global_best_pos = particle.pos().clone();
        self.global_best_pos_eval = particle.pos_eval().clone();
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

    // Initialize data storing variable.
    let mut data: Vec<(f64, Vec<f64>)> = Vec::new();

    for _ in 0..iterations {
      // Variable to store data for this iteration.
      let mut iteration_data: Vec<f64> = Vec::new();

      for particle in self.particles.iter_mut() {
        particle.update_vel(&self.global_best_pos);
        if particle.update_pos() {
          // `update_pos()` returns `true` if its personal best was updated,
          // in which case the global best should be examined as well.
          if particle.best_pos_eval() < &self.global_best_pos_eval {
            self.global_best_pos = particle.best_pos().clone();
            self.global_best_pos_eval = particle.best_pos_eval().clone();
          }
        }
        iteration_data.push(particle.pos_eval().clone());
      }

      // Save the data for current iteration.
      data.push((self.global_best_pos_eval, iteration_data));

      // Increment the progress bar.
      progress.inc(1);
    }

    // Finish the progress bar.
    progress.finish();

    // Draw and save the graph for the run.
    let _ = grapher::create_progress_graph(&data);
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
      utils::format_dvector(&self.global_best_pos),
      self.global_best_pos_eval,
    ));
    write!(f, "{}", result)
  }
}
