## kneed-rs

This is a pure rust implementation of Knee-point detection.

The code here aims to be a 1:1 match of [kneed](https://pypi.org/project/kneed/).

### Usage

Example from the paper:

```rust
let (x, y) = DataGenerator::figure2();

let params = KneeLocatorParams::new(
    ValidCurve::Concave,
    ValidDirection::Increasing,
    InterpMethod::Interp1d,
);
let kneedle = KneeLocator::new(x.to_vec(), y.to_vec(), 1.0, params);

assert_relative_eq!(0.222222222222222, kneedle.knee.unwrap());
assert_relative_eq!(1.8965517241379306, kneedle.knee_y.unwrap());
```

Using score values from a CSV:

```rust
let mut rdr = csv::Reader::from_path("/path/to/data.csv").unwrap();
let mut scores = Vec::new();
for result in rdr.records() {
    let record = result.unwrap();
    let score = record.get(1).unwrap().parse::<f64>().unwrap();
    scores.push(score)
}
scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

let range = 0..scores.len();
let x = Array1::from_iter(range.map(|i| i as f64));
let y = Array1::from_vec(scores);

let params = KneeLocatorParams::new(
    ValidCurve::Convex,
    ValidDirection::Increasing,
    InterpMethod::Interp1d,
);
let kneedle = KneeLocator::new(x, y, 1.0, params);
let elbow = kneedle.elbow().unwrap();
println!("elbow: {:?}", elbow);

let elbow_y = kneedle.elbow_y();
println!("elbow_y: {:?}", elbow_y);
```

### Credits

All credit for the python implementation goes to [Kevin Arvai](https://github.com/arvkevi).
