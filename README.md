# Usage
## Running PSO

```rust
let params: HashMap<String, ParamValue> = [
  ("w".to_owned(), ParamValue::Float(0.8)),
  ("phi_p".to_owned(), ParamValue::Float(1.0)),
  ("phi_g".to_owned(), ParamValue::Float(1.0)),
  ("particle_count".to_owned(), ParamValue::Int(30)),
]
.iter()
.cloned()
.collect();

let mut pso: PSO<Particle> = PSO::new(
  "PSO",
  functions::f1(10),
  params.clone(),
  PathBuf::from("data/test"),
);
pso.run(iterations);
pso.save_summary()?;
pso.save_data()?;
pso.save_config()?;
```

## Grid search
```rust
use grid_search::grid_search_dim;

// Experiment Settings
let iterations = 1000;
let out_directory = PathBuf::from("data/base_pso_all");

// Test particle_count vs. dimensions for CEC2017
let cec17_dims = vec![2, 10, 20, 30, 50, 100];
let particle_counts = vec![
  ParamValue::Int(2),
  ParamValue::Int(5),
  ParamValue::Int(10),
  ParamValue::Int(50),
  ParamValue::Int(100),
  ParamValue::Int(200),
  ParamValue::Int(500),
];
let base_params: HashMap<String, ParamValue> = [
  ("w".to_owned(), ParamValue::Float(0.8)),
  ("phi_p".to_owned(), ParamValue::Float(1.0)),
  ("phi_g".to_owned(), ParamValue::Float(1.0)),
]
.iter()
.cloned()
.collect();

for func_num in 1..=30 {
  if func_num == 2 {
    continue;
  }
  grid_search_dim::<Particle, PSO<Particle>>(
    iterations,
    Arc::new(move |d: usize| functions::cec17(func_num, d)),
    5,
    cec17_dims.clone(),
    ("particle_count".to_owned(), particle_counts.clone()),
    base_params.clone(),
    out_directory.clone(),
  )?;
}
```

# Running the Visualizer
```
source .venv/bin/activate
pipenv shell
python visualizer/main.py
```

# Acknowledgement
### CEC2017 Benchmark Functions
N. H. Awad, M. Z. Ali, J. J. Liang, B. Y. Qu and P. N. Suganthan, "Problem Definitions and Evaluation Criteria for the CEC 2017 Special Session and Competition on Single Objective Bound Constrained Real-Parameter Numerical Optimization," Technical Report, Nanyang Technological University, Singapore, November 2016