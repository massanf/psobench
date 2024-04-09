use crate::pso_trait::ParamValue;
use lazy_static::lazy_static;
use std::collections::HashMap;

lazy_static! {
  pub static ref PSO_PARAMS: HashMap<String, ParamValue> = {
    let pso_params: HashMap<String, ParamValue> = [
      ("w".to_owned(), ParamValue::Float(0.8)),
      ("phi_p".to_owned(), ParamValue::Float(1.0)),
      ("phi_g".to_owned(), ParamValue::Float(1.0)),
      ("particle_count".to_owned(), ParamValue::Int(30)),
    ]
    .iter()
    .cloned()
    .collect();
    pso_params
  };
  pub static ref GSA_PARAMS: HashMap<String, ParamValue> = {
    let gsa_params: HashMap<String, ParamValue> = [
      ("g0".to_owned(), ParamValue::Float(5000.0)),
      ("alpha".to_owned(), ParamValue::Float(5.0)),
      ("particle_count".to_owned(), ParamValue::Int(30)),
    ]
    .iter()
    .cloned()
    .collect();
    gsa_params
  };
  pub static ref IGSA_PARAMS: HashMap<String, ParamValue> = {
    let gsa_params: HashMap<String, ParamValue> = [
      ("g0".to_owned(), ParamValue::Float(5000.0)),
      ("alpha".to_owned(), ParamValue::Float(5.0)),
      ("particle_count".to_owned(), ParamValue::Int(30)),
    ]
    .iter()
    .cloned()
    .collect();
    gsa_params
  };
  pub static ref TILED_GSA_PARAMS: HashMap<String, ParamValue> = {
    let tiled_gsa_params: HashMap<String, ParamValue> = [
      ("g0".to_owned(), ParamValue::Float(5000.0)),
      ("alpha".to_owned(), ParamValue::Float(5.0)),
      ("particle_count".to_owned(), ParamValue::Int(30)),
    ]
    .iter()
    .cloned()
    .collect();
    tiled_gsa_params
  };
  // pub static ref PHI_P_PARAMS: (String, Vec<ParamValue>) = {
  pub static ref PSO_PHI_P_OPTIONS: (String, Vec<ParamValue>) = {
    let phi_p: Vec<ParamValue> = vec![
      ParamValue::Float(-4.0),
      ParamValue::Float(-3.0),
      ParamValue::Float(-2.0),
      ParamValue::Float(-1.0),
      ParamValue::Float(0.0),
      ParamValue::Float(1.0),
      ParamValue::Float(2.0),
      ParamValue::Float(3.0),
      ParamValue::Float(4.0),
    ];
    ("phi_p".to_owned(), phi_p)
  };
  pub static ref PSO_PHI_G_OPTIONS: (String, Vec<ParamValue>) = {
    let phi_g: Vec<ParamValue> = vec![
      ParamValue::Float(-4.0),
      ParamValue::Float(-3.0),
      ParamValue::Float(-2.0),
      ParamValue::Float(-1.0),
      ParamValue::Float(0.0),
      ParamValue::Float(1.0),
      ParamValue::Float(2.0),
      ParamValue::Float(3.0),
      ParamValue::Float(4.0),
    ];
    ("phi_g".to_owned(), phi_g)
  };
  pub static ref PSO_BASE_PARAMS: HashMap<String, ParamValue> = {
    let base_params: HashMap<String, ParamValue> = [
      ("w".to_owned(), ParamValue::Float(0.8)),
      ("particle_count".to_owned(), ParamValue::Int(30)),
    ]
    .iter()
    .cloned()
    .collect();
    base_params
  };
  pub static ref GSA_G0_OPTIONS: (String, Vec<ParamValue>) = {
    let g0: Vec<ParamValue> = vec![
      ParamValue::Float(2.0),
      ParamValue::Float(5.0),
      ParamValue::Float(10.0),
      ParamValue::Float(20.0),
      ParamValue::Float(50.0),
      ParamValue::Float(100.0),
      ParamValue::Float(200.0),
      ParamValue::Float(500.0),
      ParamValue::Float(1000.0),
      ParamValue::Float(2000.0),
      ParamValue::Float(5000.0),
      ParamValue::Float(10000.0),
    ];
    ("g0".to_owned(), g0)
  };
  pub static ref GSA_ALPHA_OPTIONS: (String, Vec<ParamValue>) = {
    let alpha: Vec<ParamValue> = vec![
      ParamValue::Float(1.0),
      ParamValue::Float(2.0),
      ParamValue::Float(5.0),
      ParamValue::Float(10.0),
      ParamValue::Float(20.0),
      ParamValue::Float(50.0),
      ParamValue::Float(100.0),
    ];
    ("alpha".to_owned(), alpha)
  };
  pub static ref GSA_BASE_PARAMS: HashMap<String, ParamValue> = {
    let base_params: HashMap<String, ParamValue> =
    [("particle_count".to_owned(), ParamValue::Int(30))].iter().cloned().collect();
    base_params
  };
}
