// use crate::Particle;
use plotters::prelude::*;
use std::path::Path;

// pub fn create_2d_scatter(particles: &Vec<Particle>) -> Result<(), Box<dyn std::error::Error>> {
//     let root = BitMapBackend::new("graphs/0.png", (640, 480)).into_drawing_area();
//     root.fill(&WHITE)?;
//     let mut chart = ChartBuilder::on(&root)
//         .margin(5)
//         .x_label_area_size(30)
//         .y_label_area_size(30)
//         .build_cartesian_2d(-1f32..1f32, -1f32..1f32)?;

//     chart.configure_mesh().draw()?;

//     chart.draw_series(particles.iter().map(|particle| {
//         Circle::new(
//             (particle.pos()[0] as f32, particle.pos()[1] as f32),
//             3,
//             ShapeStyle::from(&RED).filled(),
//         )
//     }))?;
//     root.present()?;
//     Ok(())
// }

pub fn generate_progress_graph(filename: &Path, data: &Vec<f64>) -> Result<(), Box<dyn std::error::Error>> {
  // Create the root drawing area for the graph.
  let root = BitMapBackend::new(filename, (640, 480)).into_drawing_area();

  root.fill(&WHITE)?;

  // Set the caption, margin, and the axis.
  let mut chart = ChartBuilder::on(&root)
    .margin(5)
    .x_label_area_size(30)
    .y_label_area_size(30)
    .build_cartesian_2d(0f32..data.len() as f32, (0.000001f32..1000f32).log_scale())?;

  chart.configure_mesh().draw()?;

  //   for (iteration, (_g, p)) in data.iter().enumerate() {
  //     // Draw a blue circle for each particle in (iteration, fitness).
  //     chart.draw_series(
  //       p.iter().map(|datum| Circle::new((iteration as f32, *datum as f32), 1, ShapeStyle::from(&BLUE).filled())),
  //     )?;
  //   }

  // Draw a line graph for the global minimums for each iteration.
  chart.draw_series(LineSeries::new(
    data.iter().enumerate().map(|(i, g)| (i as f32, g.clone() as f32)).collect::<Vec<_>>(),
    &RED,
  ))?;

  root.present()?;
  Ok(())
}
