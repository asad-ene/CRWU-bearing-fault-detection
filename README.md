# Bearing Fault Detection — CWRU Dataset

## Problem
Predict bearing fault type from vibration signal features using machine learning.

## Dataset
Case Western Reserve University (CWRU) Bearing Dataset
- 10 classes (3 fault types + Normal)
- 2300 samples, perfectly balanced
- Features: RMS, Kurtosis, Crest Factor, Skewness etc.

## Hyperparaters setting:
- n_estimators=200,
- criterion='entropy',
- min_samples_split=5,
- max_depth=20,
- random_state=42

## Results
| Model         | Accuracy |
|---------------|----------|
| Random Forest | 96%      |
| CNN (coming)  | TBD      |

## Key Finding
Model achieve high accuracy with normal vs faulty bearings. Confusion occur only in small damaging fauls (0.007 inch) across different fault type 

## Next Steps
CNN on raw vibration signals to capture characteristic defect frequencies and improve small fault discrimination.
