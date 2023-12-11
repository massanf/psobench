extern crate nalgebra as na;
extern crate rand;
use nalgebra::DVector;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::fmt;

struct Particle {
    pos: DVector<f64>,
    pos_eval: f64,
    vel: DVector<f64>,
    best_pos: DVector<f64>,
    best_pos_eval: f64,
}

impl Particle {
    fn new(dimensions: usize) -> Particle {
        let b_lo: DVector<f64> = DVector::from_element(dimensions, -1.0);
        let b_up: DVector<f64> = DVector::from_element(dimensions, 1.0);

        let pos: DVector<f64> = uniform_distribution(&b_lo, &b_up);
        let vel: DVector<f64> = uniform_distribution(
            &DVector::from_iterator(dimensions, (&b_up - &b_lo).iter().map(|b| -b.abs())),
            &DVector::from_iterator(dimensions, (&b_up - &b_lo).iter().map(|b| b.abs())),
        );

        Particle {
            pos: pos.clone(),
            pos_eval: 0.,
            vel: vel,
            best_pos: pos,
            best_pos_eval: f64::MAX,
        }
    }

    fn update_vel(&mut self, global_best_pos: &DVector<f64>) {
        let w = 0.8;
        let phi_p = 2.;
        let phi_g = 2.;
        let mut rng = rand::thread_rng();
        let r_p: f64 = rng.gen_range(0.0..1.0);
        let r_g: f64 = rng.gen_range(0.0..1.0);
        self.vel = w * &self.vel
            + phi_p * r_p * (&self.best_pos - &self.pos)
            + phi_g * r_g * (global_best_pos - &self.pos);
    }

    fn update_pos(&mut self) -> bool {
        // Returns whether the best was updated.
        self.pos = &self.pos + &self.vel;
        self.eval()
    }

    fn eval(&mut self) -> bool {
        // Returns whether the best was updated.
        self.pos_eval = self.pos.iter().map(|x| x.powi(2)).sum();
        if self.pos_eval < self.best_pos_eval {
            self.best_pos = self.pos.clone();
            self.best_pos_eval = self.pos_eval.clone();
            return true;
        }
        false
    }
}

impl fmt::Display for Particle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            &format!(
                " pos:  [{}] ({:.3})\n vel:  [{}]\n best: [{}] ({:.3})\n",
                format_dvector(&self.pos),
                self.pos_eval,
                format_dvector(&self.vel),
                format_dvector(&self.best_pos),
                self.best_pos_eval,
            )
        )
    }
}

struct PSO {
    particles: Vec<Particle>,
    global_best_pos: DVector<f64>,
    global_best_pos_eval: f64,
}

impl PSO {
    fn new(dimensions: usize, number_of_particles: usize) -> PSO {
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

    fn init(&mut self) {
        for particle in &self.particles {
            if particle.pos_eval < self.global_best_pos_eval {
                self.global_best_pos = particle.pos.clone();
                self.global_best_pos_eval = particle.pos_eval.clone();
            }
        }
    }

    fn run(&mut self, iterations: usize) {
        for _ in 0..iterations {
            for particle in self.particles.iter_mut() {
                particle.update_vel(&self.global_best_pos);
                if particle.update_pos() {
                    if particle.best_pos_eval < self.global_best_pos_eval {
                        self.global_best_pos = particle.best_pos.clone();
                        self.global_best_pos_eval = particle.best_pos_eval.clone();
                    }
                }
            }
        }
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
            format_dvector(&self.global_best_pos),
            self.global_best_pos_eval,
        ));
        write!(f, "{}", result)
    }
}

fn uniform_distribution(low: &DVector<f64>, high: &DVector<f64>) -> DVector<f64> {
    let mut rng = rand::thread_rng();
    DVector::from_iterator(
        low.len(),
        (0..low.len()).map(|i| Uniform::new(low[i], high[i]).sample(&mut rng)),
    )
}

fn format_dvector(vec: &DVector<f64>) -> String {
    let mut result = String::new();
    for i in 0..vec.len() {
        result.push_str(&format!("{:.3}", vec[i]));
        if i != vec.len() - 1 {
            result.push_str(", ");
        }
    }
    result
}

fn main() {
    let mut pso = PSO::new(3, 300);
    pso.init();
    pso.run(1000);
    println!("{}", pso);
}
