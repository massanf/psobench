use nalgebra::DVector;
use rand::distributions::{Distribution, Uniform};

pub fn uniform_distribution(low: &DVector<f64>, high: &DVector<f64>) -> DVector<f64> {
    let mut rng = rand::thread_rng();
    DVector::from_iterator(
        low.len(),
        (0..low.len()).map(|i| Uniform::new(low[i], high[i]).sample(&mut rng)),
    )
}

pub fn format_dvector(vec: &DVector<f64>) -> String {
    let mut result = String::new();
    for i in 0..vec.len() {
        result.push_str(&format!("{:.3}", vec[i]));
        if i != vec.len() - 1 {
            result.push_str(", ");
        }
    }
    result
}
