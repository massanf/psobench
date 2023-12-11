extern crate nalgebra as na;

use nalgebra::DVector;
use std::fmt;

use crate::rand::Rng;
use crate::utils;

pub struct Particle {
    pos: DVector<f64>,
    pos_eval: f64,
    vel: DVector<f64>,
    best_pos: DVector<f64>,
    best_pos_eval: f64,
}

impl Particle {
    pub fn new(dimensions: usize) -> Particle {
        let b_lo: DVector<f64> = DVector::from_element(dimensions, -1.0);
        let b_up: DVector<f64> = DVector::from_element(dimensions, 1.0);

        let pos: DVector<f64> = utils::uniform_distribution(&b_lo, &b_up);
        let vel: DVector<f64> = utils::uniform_distribution(
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

    pub fn pos(&mut self) -> &DVector<f64> {
        &self.pos
    }

    pub fn pos_eval(&mut self) -> &f64 {
        &self.pos_eval
    }

    pub fn best_pos(&mut self) -> &DVector<f64> {
        &self.best_pos
    }

    pub fn best_pos_eval(&mut self) -> &f64 {
        &self.best_pos_eval
    }

    pub fn update_vel(&mut self, global_best_pos: &DVector<f64>) {
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

    pub fn update_pos(&mut self) -> bool {
        // Returns whether the best was updated.
        self.pos = &self.pos + &self.vel;
        self.eval()
    }

    pub fn eval(&mut self) -> bool {
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
                utils::format_dvector(&self.pos),
                self.pos_eval,
                utils::format_dvector(&self.vel),
                utils::format_dvector(&self.best_pos),
                self.best_pos_eval,
            )
        )
    }
}
