use anyhow::Result;
use ndarray::{s, Array1, Axis};
use ndarray_interp::interp1d::Interp1DBuilder;
use ndarray_stats::QuantileExt;
use polyfit_rs::polyfit_rs::polyfit;

#[derive(Debug, PartialEq)]
pub enum ValidCurve {
    Convex,
    Concave,
}

#[derive(Debug, PartialEq)]
pub enum ValidDirection {
    Increasing,
    Decreasing,
}

#[derive(Debug, PartialEq)]
pub enum InterpMethod {
    Interp1d,
    Polynomial,
}

#[derive(Debug)]
pub struct KneeLocatorParams {
    curve: ValidCurve,
    direction: ValidDirection,
    interp_method: InterpMethod,
}

impl KneeLocatorParams {
    pub fn new(curve: ValidCurve, direction: ValidDirection, interp_method: InterpMethod) -> Self {
        Self {
            curve,
            direction,
            interp_method,
        }
    }
}

#[derive(Debug)]
pub struct KneeLocator {
    x: Array1<f64>,
    y: Array1<f64>,
    curve: ValidCurve,
    direction: ValidDirection,
    s: f64,
    n: usize,
    all_knees: Vec<f64>,
    all_norm_knees: Vec<f64>,
    all_knees_y: Vec<f64>,
    all_norm_knees_y: Vec<f64>,
    online: bool,
    polynomial_degree: usize,
    x_normalized: Array1<f64>,
    y_normalized: Array1<f64>,
    y_difference: Array1<f64>,
    x_difference: Array1<f64>,
    maxima_indices: Vec<usize>,
    minima_indices: Vec<usize>,
    x_difference_maxima: Array1<f64>,
    y_difference_maxima: Array1<f64>,
    x_difference_minima: Array1<f64>,
    y_difference_minima: Array1<f64>,
    tmx: Array1<f64>,
    knee: Option<f64>,
    norm_knee: Option<f64>,
    knee_y: Option<f64>,
    norm_knee_y: Option<f64>,
}

impl KneeLocator {
    pub fn new(
        x: Array1<f64>,
        y: Array1<f64>,
        s: f64,
        params: KneeLocatorParams,
        online: bool,
        polynomial_degree: usize,
    ) -> Self {
        let n = x.len();
        let mut knee_locator = KneeLocator {
            x,
            y,
            curve: params.curve,
            direction: params.direction,
            s,
            n,
            all_knees: Vec::new(),
            all_norm_knees: Vec::new(),
            all_knees_y: Vec::new(),
            all_norm_knees_y: Vec::new(),
            online,
            polynomial_degree,
            x_normalized: Array1::zeros(n),
            y_normalized: Array1::zeros(n),
            y_difference: Array1::zeros(n),
            x_difference: Array1::zeros(n),
            y_difference_maxima: Array1::zeros(n),
            x_difference_maxima: Array1::zeros(n),
            y_difference_minima: Array1::zeros(n),
            x_difference_minima: Array1::zeros(n),
            maxima_indices: Vec::new(),
            minima_indices: Vec::new(),
            tmx: Array1::zeros(0),
            knee: None,
            norm_knee: None,
            knee_y: None,
            norm_knee_y: None,
        };

        knee_locator.initialize(params.interp_method);
        knee_locator
    }

    fn initialize(&mut self, interp_method: InterpMethod) {
        // Step 1: Fit a smooth line
        let ds_y = match interp_method {
            InterpMethod::Interp1d => self.interp1d().unwrap(),
            InterpMethod::Polynomial => self.polynomial_interp().unwrap(),
        };

        // Step 2: Normalize values
        self.x_normalized = Self::normalize(&self.x).unwrap();
        self.y_normalized = Self::normalize(&ds_y).unwrap();

        // Step 3: Calculate the difference curve
        self.y_normalized = self.transform_y();
        self.y_difference = &self.y_normalized - &self.x_normalized;
        self.x_difference.clone_from(&self.x_normalized);

        // Step 4: Identify local maxima/minima
        self.find_local_extrema();

        // Step 5: Calculate thresholds
        self.calculate_thresholds();

        // Step 6: Find knee
        (self.knee, self.norm_knee) = self.find_knee();

        // Step 7: If we have a knee, extract data about it
        self.knee_y = match self.knee {
            None => None,
            Some(knee) => Some(self.y[self.x.iter().position(|&v| v == knee).unwrap()]),
        };

        self.norm_knee_y = match self.norm_knee {
            None => None,
            Some(norm_knee) => Some(
                self.y_normalized[self
                    .x_normalized
                    .iter()
                    .position(|&v| v == norm_knee)
                    .unwrap()],
            ),
        };
    }

    fn normalize(a: &Array1<f64>) -> Result<Array1<f64>> {
        let min = a.min()?;
        let max = a.max()?;
        Ok((a - *min) / (*max - *min))
    }

    fn interp1d(&self) -> Result<Array1<f64>> {
        let interpolator = Interp1DBuilder::new(self.y.view())
            .x(self.x.view())
            .build()?;

        Ok(Array1::from_vec(
            self.x
                .iter()
                .map(|&x| interpolator.interp_scalar(x).unwrap())
                .collect(),
        ))
    }

    fn polynomial_interp(&self) -> Result<Array1<f64>> {
        let coeffs = polyfit(
            self.x.as_slice().unwrap(),
            self.y.as_slice().unwrap(),
            self.polynomial_degree,
        )
        .map_err(|e| anyhow::anyhow!(e))?;

        Ok(Array1::from_vec(
            self.x
                .iter()
                .map(|&x| {
                    coeffs.iter().enumerate().fold(0.0, |acc, (power, &coeff)| {
                        acc + coeff * x.powi(power as i32)
                    })
                })
                .collect(),
        ))
    }

    fn transform_y(&self) -> Array1<f64> {
        match (&self.direction, &self.curve) {
            (ValidDirection::Decreasing, ValidCurve::Concave) => {
                self.y_normalized.slice(s![..;-1]).to_owned()
            }
            (ValidDirection::Decreasing, ValidCurve::Convex) => {
                let max = self.y_normalized.max().unwrap();
                self.y_normalized.mapv(|v| max - v)
            }
            (ValidDirection::Increasing, ValidCurve::Convex) => {
                let max = self.y_normalized.max().unwrap();
                self.y_normalized
                    .mapv(|v| max - v)
                    .slice(s![..;-1])
                    .to_owned()
            }
            _ => self.y_normalized.clone(), // No transformation needed for (Increasing, Concave)
        }
    }

    fn find_local_extrema(&mut self) {
        // Local maxima
        self.maxima_indices = argrelextrema(&self.y_difference, |a, b| a >= b, 1);
        self.x_difference_maxima = self
            .maxima_indices
            .iter()
            .map(|&i| self.x_difference[i])
            .collect::<Vec<f64>>()
            .into();
        self.y_difference_maxima = self
            .maxima_indices
            .iter()
            .map(|&i| self.y_difference[i])
            .collect::<Vec<f64>>()
            .into();

        // Local minima
        self.minima_indices = argrelextrema(&self.y_difference, |a, b| a <= b, 1);
        self.x_difference_minima = self
            .minima_indices
            .iter()
            .map(|&i| self.x_difference[i])
            .collect::<Vec<f64>>()
            .into();
        self.y_difference_minima = self
            .minima_indices
            .iter()
            .map(|&i| self.y_difference[i])
            .collect::<Vec<f64>>()
            .into();
    }

    fn find_local_maxima(&self, arr: &Array1<f64>) -> Vec<usize> {
        let mut maxima = Vec::new();
        for i in 1..arr.len() - 1 {
            if arr[i] >= arr[i - 1] && arr[i] >= arr[i + 1] {
                maxima.push(i);
            }
        }
        maxima
    }

    fn find_local_minima(&self, arr: &Array1<f64>) -> Vec<usize> {
        let mut minima = Vec::new();
        for i in 1..arr.len() - 1 {
            if arr[i] <= arr[i - 1] && arr[i] <= arr[i + 1] {
                minima.push(i);
            }
        }
        minima
    }

    fn calculate_thresholds(&mut self) {
        let mean_diff = self
            .x_normalized
            .windows(2)
            .into_iter()
            .map(|w| w[1] - w[0])
            .fold(0.0, |acc, x| acc + x)
            / (self.n - 1) as f64;

        let selected_y_diff = self.y_difference.select(Axis(0), &self.maxima_indices);

        self.tmx = &selected_y_diff - (self.s * mean_diff.abs());
    }

    pub fn find_knee(&mut self) -> (Option<f64>, Option<f64>) {
        // Return None if no local maxima found
        if self.maxima_indices.is_empty() {
            return (None, None);
        }

        // Placeholders for which threshold region i is located in
        let mut maxima_threshold_index = 0;
        let mut minima_threshold_index = 0;
        let mut knee: Option<f64> = None;
        let mut norm_knee: Option<f64> = None;
        let mut threshold = 0.0;
        let mut threshold_index = 0;

        // Traverse the difference curve
        for (i, &_x) in self.x_difference.iter().enumerate() {
            // Skip points on the curve before the first local maxima
            if i < self.maxima_indices[0] {
                continue;
            }

            let j = i + 1;

            // Reached the end of the curve
            if i == (self.x_difference.len() - 1) {
                break;
            }

            // If we're at a local max, increment the maxima threshold index and continue
            if self.maxima_indices.contains(&i) {
                threshold = self.tmx[maxima_threshold_index];
                threshold_index = i;
                maxima_threshold_index += 1;
            }

            // Values in difference curve are at or after a local minimum
            if self.minima_indices.contains(&i) {
                threshold = 0.0;
                minima_threshold_index += 1;
            }

            if self.y_difference[j] < threshold {
                match self.curve {
                    ValidCurve::Convex => {
                        if self.direction == ValidDirection::Decreasing {
                            knee = Some(self.x[threshold_index]);
                            norm_knee = Some(self.x_normalized[threshold_index]);
                        } else {
                            knee = Some(self.x[self.x.len() - threshold_index - 1]);
                            norm_knee = Some(self.x_normalized[threshold_index]);
                        }
                    }
                    ValidCurve::Concave => {
                        if self.direction == ValidDirection::Decreasing {
                            knee = Some(self.x[self.x.len() - threshold_index - 1]);
                            norm_knee = Some(self.x_normalized[threshold_index]);
                        } else {
                            knee = Some(self.x[threshold_index]);
                            norm_knee = Some(self.x_normalized[threshold_index]);
                        }
                    }
                }

                // Add the y value at the knee
                let y_at_knee = self.y[self.x.iter().position(|&v| v == knee.unwrap()).unwrap()];
                let y_norm_at_knee = self.y_normalized[self
                    .x_normalized
                    .iter()
                    .position(|&v| v == norm_knee.unwrap())
                    .unwrap()];

                if !self.all_knees.contains(&knee.unwrap()) {
                    self.all_knees_y.push(y_at_knee);
                    self.all_norm_knees_y.push(y_norm_at_knee);
                }

                self.all_knees.push(knee.unwrap());
                self.all_norm_knees.push(norm_knee.unwrap());

                // If detecting in offline mode, return the first knee found
                if !self.online {
                    return (knee, norm_knee);
                }
            }
        }

        if self.all_knees.is_empty() {
            // No knee was found
            return (None, None);
        }

        (knee, norm_knee)
    }

    pub fn elbow(&self) -> Option<f64> {
        self.knee
    }

    pub fn norm_elbow(&self) -> Option<f64> {
        self.norm_knee
    }

    pub fn elbow_y(&self) -> Option<f64> {
        self.knee_y
    }

    pub fn norm_elbow_y(&self) -> Option<f64> {
        self.norm_knee_y
    }
}

// Re-implement argrelextrema from numpy
fn argrelextrema<F>(data: &Array1<f64>, comparator: F, order: usize) -> Vec<usize>
where
    F: Fn(f64, f64) -> bool,
{
    let mut extrema_indices = Vec::new();
    let len = data.len();

    for i in 0..len {
        let mut is_extrema = true;

        // Compare with previous `order` elements
        for j in 1..=order {
            if i >= j && !comparator(data[i], data[i - j]) {
                is_extrema = false;
                break;
            }
        }

        // Compare with next `order` elements
        for j in 1..=order {
            if i + j < len && !comparator(data[i], data[i + j]) {
                is_extrema = false;
                break;
            }
        }

        if is_extrema {
            extrema_indices.push(i);
        }
    }

    extrema_indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_generator::DataGenerator;
    use approx::assert_relative_eq;

    #[test]
    fn test_known() {
        let (x, y) = DataGenerator::figure2();

        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Increasing,
            InterpMethod::Interp1d,
        );
        let kneedle = KneeLocator::new(x, y, 1.0, params, false, 7);

        assert_relative_eq!(0.222222222222222, kneedle.knee.unwrap());
        assert_relative_eq!(1.8965517241379306, kneedle.knee_y.unwrap());
    }

    // #[test]
    // fn test_known_csv_values() {
    //     let mut rdr = csv::Reader::from_path("/path/to/csv").unwrap();
    //     let mut scores = Vec::new();
    //     for result in rdr.records() {
    //         let record = result.unwrap();
    //         let score = record.get(1).unwrap().parse::<f64>().unwrap();
    //         scores.push(score)
    //     }
    //     scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    //
    //     let range = 0..scores.len();
    //     let x = Array1::from_iter(range.map(|i| i as f64));
    //     let y = Array1::from_vec(scores);
    //
    //     let params = KneeLocatorParams::new(
    //         ValidCurve::Convex,
    //         ValidDirection::Increasing,
    //         InterpMethod::Interp1d,
    //     );
    //     let kneedle = KneeLocator::new(x, y, 1.0, params, false, 7);
    //     let elbow = kneedle.elbow().unwrap();
    //     println!("elbow: {:?}", elbow);
    //
    //     let elbow_y = kneedle.elbow_y();
    //     println!("elbow_y: {:?}", elbow_y);
    //
    //     assert!(false)
    // }
}
