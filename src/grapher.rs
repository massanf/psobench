// use crate::Particle;
use crate::particle;
use plotters::prelude::*;

use particle::Particle;

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

pub fn create_progress_graph(
    data: &Vec<(f64, Vec<f64>)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let max = data
        .iter()
        .flat_map(|(_, vec)| vec.iter())
        .fold(f64::MIN, |acc, &val| acc.max(val));

    let root = BitMapBackend::new("graphs/0.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0f32..data.len() as f32, 0f32..0.01f32)?;

    chart.configure_mesh().draw()?;

    for (i, (_g, p)) in data.iter().enumerate() {
        chart.draw_series(p.iter().map(|datum| {
            Circle::new(
                (i as f32, *datum as f32),
                1,
                ShapeStyle::from(&BLUE).filled(),
            )
        }))?;
    }

    chart.draw_series(LineSeries::new(
        data.iter()
            .enumerate()
            .map(|(i, (g, _p))| (i as f32, g.clone() as f32))
            .collect::<Vec<_>>(),
        &RED,
    ))?;

    root.present()?;
    Ok(())
}
