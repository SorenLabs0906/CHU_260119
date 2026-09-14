# Data format and preprocessing

## Source recordings

`data/raw/train set/` contains three sensor exports: `imudata.csv`, `rtk_lla.csv` and `gnss.csv`. Each test directory contains the corresponding `_labeled.csv` and `_mask.csv` files plus its `annotations.json` file. These are the selected CSV exports supplied with the dataset; “raw” does not mean unprocessed binary receiver packets.

The complete file inventory, row counts, headers and checksums are in [raw_files.json](raw_files.json).

| Stream | Nominal rate | Core fields | Role |
|---|---:|---|---|
| `rtk_lla` | 400 Hz | `timestamp, latitude, longitude, altitude` | High-rate position observation used to build the position-increment channels |
| `imudata` | 400 Hz | `timestamp`, quaternion, angular velocity, acceleration | Six inertial channels enter the processed representation; quaternion is retained only in the source files |
| `gnss` | 4 Hz | `timestamp`, position, status and reported covariance | Separate receiver output; retained for downstream studies, not used by the nine-channel preprocessing |

Timestamps are Unix seconds. Latitude and longitude are in degrees; altitude is in metres. The preprocessing uses the supplied altitude directly with the WGS84 coordinate conversion and does not apply a geoid correction. IMU components retain the recorded sensor axes; no mounting-axis interpretation, gravity removal or orientation-frame convention is added by this release. Auxiliary sequence numbers, frame identifiers and device status/covariance fields are preserved as recorded.

The raw sensor CSVs can have different lengths and millisecond timing offsets. Treat timestamps as the alignment authority. `_labeled.csv` suffixes indicate the presence of annotation columns, not independent clean ground truth.

## Processed files

Each `data/processed/<subset>/` directory contains the following files.

| File | Shape or schema | Interpretation |
|---|---|---|
| `features.csv` | N rows, 9 named columns | Physical-unit observation values, decimal precision of 9 digits after the decimal point |
| `timestamps.npy` | `(N,)`, float64 | Processed timeline, Unix seconds |
| `positions_enu.npy` | `(N,3)`, float64 | Position observations in local East, North, Up coordinates, metres |
| `anomaly_mask.npy` | `(N,9)`, int8 | 1 = annotated anomalous; 0 = not annotated anomalous |
| `binary_labels.npy` | `(N,)`, int8 | Logical OR of the nine channel masks |
| `source_labels.npy` | `(N,)`, int8 | 0 normal, 1 GNSS, 2 IMU; joint source labels are absent |
| `events.csv` | Variable number of rows | Maximal contiguous intervals of each nonzero source label |
| `metadata.json` | JSON object | Counts, units, source identity, coordinate origin and preprocessing profile |

All NPY files are numeric and load with `allow_pickle=False`. Do not interpret a filename from a previous experimental layout as a semantic definition: this release uses `features.csv` for observations, including corrupted observations in the test data.

Channel order is `[dE, dN, dU, ax, ay, az, gx, gy, gz]`. GNSS labels apply to channels 0–2, and IMU labels to channels 3–8. The mask has the same order as the feature columns. Acceleration and angular-velocity source masks retain their separate channel positions.

`events.csv` uses `start_index` (included) and `end_index_exclusive` (excluded). Thus `features[start_index:end_index_exclusive]` selects the event. The end timestamp is the next frame's timestamp, or one median sampling interval after the last frame when an event reaches the end. Event frame counts are authoritative; a printed decimal duration may differ slightly from frame count / 200 because timestamps retain recording jitter.

Raw annotation JSONs describe sensor annotation intervals. Their acceleration and gyroscope intervals may overlap in time. The processed source-level event list merges overlapping or adjacent IMU-labelled frames; it is therefore not a count of all raw JSON segments.

## Coordinate conversion

Longitude, latitude and altitude are transformed through WGS84 Earth-centred coordinates to ENU, using the exact per-record reference in [preprocessing.json](preprocessing.json). Each recording has its own reference origin. Do not overlay different recordings' ENU positions without transforming them into a common reference frame.

`dE, dN, dU` are consecutive differences of the reduced position stream, with the first vector set to zero. The processed absolute positions are preserved separately, because summing decimal-rounded increments over a long recording can introduce a small rounding error.

## Reproduction profile

1. Parse and sort timestamps. Restrict IMU observations to the shared IMU/RTK time range.
2. Match RTK positions to the IMU timeline with nearest-neighbour matching and a 0.005 s tolerance.
3. Use the exact coordinate origin in the configuration to obtain ENU positions.
4. Retain complete pairs of samples; discard a final unmatched sample if present. Apply the following reductions:

| Subset | Position reduction | Timestamp reduction | IMU reduction |
|---|---|---|---|
| `train set` | Pairwise mean | Pairwise mean | Pairwise mean |
| `test set 1` | First sample in each pair | First sample in each pair | Pairwise mean |
| `test set 2` | First sample in each pair | First sample in each pair | Pairwise mean |
| `test set 3` | First sample in each pair | First sample in each pair | Pairwise mean |

5. Difference the reduced positions. No normalization, learned model or anomaly repair is applied.
6. Align the annotation masks to the same retained IMU timeline. OR the three RTK coordinate masks and replicate this group label across `dE, dN, dU`; retain the six IMU mask columns. Downsample masks by pairwise OR.
7. Derive the binary label and mutually exclusive source label. No short-event filtering, gap filling or detector-based relabelling is added.

For a stride-position test frame k, the position timestamp is that of the first original sample in its pair, whereas the associated IMU vector is the average of both samples. This timing convention is retained for reproducibility. The conversion is an offline preprocessing procedure, not a claim of strictly causal online availability.

The inherited alignment fallback forward/back-fills missing RTK matches and assigns zero to missing annotation matches. For all four released recordings, the verified unmatched-row counts are **zero**, so neither fallback changes the data. See per-subset metadata for those diagnostics.

## Normalization and windows

The processed release contains full recordings, not overlapping model windows. There is no required window length, model-specific scaler or fixed internal validation split. Fit any scaler on the chosen normal training portion only, state its definition, and apply it unchanged to validation/test data. Record window length, stride, padding and handling of incomplete windows when reporting results.

For recovery experiments, a ground-truth channel anomaly mask identifies an evaluation target. It should not be silently presented as the output of an anomaly detector. The inverse mask `1 - anomaly_mask` is a keep mask only for methods with that convention.

## Interpretation limits

The processed position file and the RTK source CSV derive from the same measured stream. They are not independent ground truth within an anomalous GNSS interval. The release does not include an independent clean reconstruction target for these intervals, trained model weights, detector thresholds, or model-comparison results.

`source_labels` describe annotation categories. Their mutual exclusivity is a property of this annotation protocol; it does not imply that simultaneous physical disturbances are impossible or that unlabelled channels are universally error-free.
