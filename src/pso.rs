use crate::grapher;
use crate::particle;
use crate::utils;

use nalgebra::DVector;
use particle::Particle;
use std::fmt;

pub struct PSO {
    particles: Vec<Particle>,
    global_best_pos: DVector<f64>,
    global_best_pos_eval: f64,
}

impl PSO {
    pub fn new(dimensions: usize, number_of_particles: usize) -> PSO {
        let mut particles = Vec::new();

        for _ in 0..number_of_particles {
            let mut particle = Particle::new(dimensions);
            particle.eval();
            particles.push(particle);
        }

        PSO {
            particles,
            global_best_pos: DVector::from_element(dimensions, -1.),
            global_best_pos_eval: f64::MAX,
        }
    }

    pub fn init(&mut self) {
        for particle in &mut self.particles {
            if particle.pos_eval() < &self.global_best_pos_eval {
                self.global_best_pos = particle.pos().clone();
                self.global_best_pos_eval = particle.pos_eval().clone();
            }
        }
    }

    pub fn run(&mut self, iterations: usize) {
        let mut data: Vec<(f64, Vec<f64>)> = Vec::new();
        for _ in 0..iterations {
            let mut iteration_data: Vec<f64> = Vec::new();
            for particle in self.particles.iter_mut() {
                particle.update_vel(&self.global_best_pos);
                if particle.update_pos() {
                    if particle.best_pos_eval() < &self.global_best_pos_eval {
                        self.global_best_pos = particle.best_pos().clone();
                        self.global_best_pos_eval = particle.best_pos_eval().clone();
                    }
                }
                iteration_data.push(particle.pos_eval().clone());
            }
            data.push((self.global_best_pos_eval, iteration_data));
            // let _ = create_2d_scatter(&self.particles);
            // thread::sleep(Duration::from_millis(500));
        }
        let _ = grapher::create_progress_graph(&data);
    }
}

impl fmt::Display for PSO {
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
