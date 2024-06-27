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
            InterpMethod::Interp1d => self.interp1d(),
            InterpMethod::Polynomial => self.polynomial_interp(),
        };

        // Step 2: Normalize values
        self.x_normalized = Self::normalize(&self.x);
        self.y_normalized = Self::normalize(&ds_y.unwrap());

        // Step 3: Calculate the difference curve
        self.y_normalized = self.transform_y();
        self.y_difference = &self.y_normalized - &self.x_normalized;
        self.x_difference.clone_from(&self.x_normalized);

        // Step 4: Identify local maxima/minima
        self.find_local_extrema();

        // Step 5: Calculate thresholds
        self.calculate_thresholds();

        // Step 6: Find knee
        self.find_knee();
    }

    fn normalize(a: &Array1<f64>) -> Array1<f64> {
        let min = a.min().unwrap();
        let max = a.max().unwrap();
        (a - *min) / (*max - *min)
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
        self.maxima_indices = self.find_extrema(true);
        self.minima_indices = self.find_extrema(false);
    }

    fn find_extrema(&self, find_maxima: bool) -> Vec<usize> {
        let mut extrema = Vec::new();
        let n = self.y_difference.len();
        for i in 1..n - 1 {
            let prev = self.y_difference[i - 1];
            let curr = self.y_difference[i];
            let next = self.y_difference[i + 1];
            if find_maxima {
                if curr >= prev && curr >= next {
                    extrema.push(i);
                }
            } else if curr <= prev && curr <= next {
                extrema.push(i);
            }
        }
        extrema
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

    fn find_knee(&mut self) {
        if self.maxima_indices.is_empty() {
            return;
        }

        let mut maxima_threshold_index = 0;
        let mut minima_threshold_index = 0;

        for i in self.maxima_indices[0]..self.x_difference.len() {
            if i == self.x_difference.len() - 1 {
                break;
            }

            let j = i + 1;
            let mut threshold = 0.0;
            let mut threshold_index = i;

            if self.maxima_indices.contains(&i) {
                threshold = self.tmx[maxima_threshold_index];
                threshold_index = i;
                maxima_threshold_index += 1;
            }

            if self.minima_indices.contains(&i) {
                threshold = 0.0;
                minima_threshold_index += 1;
            }

            if self.y_difference[j] < threshold {
                let (knee, norm_knee) = self.calculate_knee(threshold_index);
                self.update_knees(knee, norm_knee);

                if !self.online {
                    break;
                }
            }
        }
    }

    fn calculate_knee(&self, threshold_index: usize) -> (f64, f64) {
        match (&self.curve, &self.direction) {
            (ValidCurve::Convex, ValidDirection::Decreasing) => {
                (self.x[threshold_index], self.x_normalized[threshold_index])
            }
            (ValidCurve::Convex, ValidDirection::Increasing) => (
                self.x[self.n - threshold_index - 1],
                self.x_normalized[threshold_index],
            ),
            (ValidCurve::Concave, ValidDirection::Decreasing) => (
                self.x[self.n - threshold_index - 1],
                self.x_normalized[threshold_index],
            ),
            (ValidCurve::Concave, ValidDirection::Increasing) => {
                (self.x[threshold_index], self.x_normalized[threshold_index])
            }
        }
    }

    fn update_knees(&mut self, knee: f64, norm_knee: f64) {
        let y_at_knee = self.y[self.x.iter().position(|&x| x == knee).unwrap()];
        let y_norm_at_knee = self.y_normalized[self
            .x_normalized
            .iter()
            .position(|&x| x == norm_knee)
            .unwrap()];

        if !self.all_knees.contains(&knee) {
            self.all_knees_y.push(y_at_knee);
            self.all_norm_knees_y.push(y_norm_at_knee);
        }

        self.all_knees.push(knee);
        self.all_norm_knees.push(norm_knee);

        self.knee = Some(knee);
        self.norm_knee = Some(norm_knee);
        self.knee_y = Some(y_at_knee);
        self.norm_knee_y = Some(y_norm_at_knee);
    }

    pub fn elbow(&self) -> f64 {
        self.knee.unwrap()
    }

    pub fn norm_elbow(&self) -> f64 {
        self.norm_knee.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_generator::DataGenerator;
    use approx::assert_relative_eq;

    #[test]
    fn test_figure2_interp1d() {
        test_figure2(InterpMethod::Interp1d);
    }

    #[test]
    fn test_figure2_polynomial() {
        test_figure2(InterpMethod::Polynomial);
    }

    fn test_figure2(interp_method: InterpMethod) {
        let (x, y) = DataGenerator::figure2();
        let params = KneeLocatorParams::new(
            ValidCurve::Concave,
            ValidDirection::Increasing,
            interp_method,
        );
        let kl = KneeLocator::new(x, y, 1.0, params, false, 7);
        assert_relative_eq!(kl.knee.unwrap(), 0.22, epsilon = 0.05);
        assert_relative_eq!(kl.elbow(), 0.22, epsilon = 0.05);
        assert_relative_eq!(kl.norm_elbow(), kl.knee.unwrap(), epsilon = 0.05);
    }
}
