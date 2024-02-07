use crate::function;
use crate::particle::ParticleTrait;
use crate::pso::PSOTrait;
use function::OptimizationProblem;
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::DVector;
extern crate rand;
use rand::Rng;

const K1: f64 = 0.5;
const K2: f64 = 0.3;
const KP: f64 = 0.45;
const KI: f64 = 0.01;
const E: f64 = 1000.0;
const POPSIZE_SET: usize = 200;

pub struct PPPSO<'a, T: ParticleTrait> {
  name: String,
  problem: &'a OptimizationProblem,
  particles: Vec<T>,
  global_best_pos: Option<DVector<f64>>,
  data: Vec<(f64, Vec<T>)>,
  accumulated_e: isize,
}

impl<'a, T: ParticleTrait> PSOTrait<'a, T> for PPPSO<'a, T> {
  fn new(name: &str, problem: &'a OptimizationProblem, dimensions: usize, number_of_particles: usize) -> PPPSO<'a, T> {
    let mut particles: Vec<T> = Vec::new();

    for _ in 0..number_of_particles {
      particles.push(ParticleTrait::new(&problem, dimensions));
    }

    let mut pso = PPPSO {
      name: name.to_owned(),
      problem,
      particles,
      global_best_pos: None,
      data: Vec::new(),
      accumulated_e: 0,
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

    let mut rng = rand::thread_rng();

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

      let mut eliminated = Vec::new();
      for i in 0..self.particles().len() {
        if self.particles()[i].pos().clone() == global_best_pos {
          continue;
        }
        if self.particles()[i].vel().norm() > E {
          continue;
        }
        if rng.gen::<f64>() > K1 {
          continue;
        }
        if rng.gen::<f64>() < K2 {
          eliminated.push(i);
        } else {
          let r = rng.gen_range(0..self.particles().len());
          let p_r = self.particles()[r].vel().clone();
          let problem = &self.problem().clone();
          let dim = self.global_best_pos().len();
          self.particles[i].init(problem, dim);
          self.particles[i].new_pos(p_r, problem);
        }
      }

      // Actually remove eliminated particles.
      eliminated.sort_unstable_by(|a, b| b.cmp(a));
      for &index in &eliminated {
        self.particles.remove(index);
      }

      // Generate new particles.
      let popsize_k = self.particles().len();
      let e = POPSIZE_SET as isize - popsize_k as isize;
      let popsize_add = (KP * e as f64 + KI * self.accumulated_e as f64).floor() as usize;

      for _ in 0..popsize_add {
        self.particles.push(ParticleTrait::new(&self.problem, self.global_best_pos().len()));
      }

      // Accumulate the `e` value.
      self.accumulated_e += e;

      // Save the data for current iteration.
      self.add_data();

      // Increment the progress bar.
      progress.inc(1);
    }

    // Finish the progress bar.
    progress.finish();
  }
}
