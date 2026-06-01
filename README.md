# Bearing Fault Detection — CWRU Dataset

## Problem
Predict bearing fault type from vibration signal features using machine learning.

## Dataset
Case Western Reserve University (CWRU) Bearing Dataset
- 10 classes (3 fault types + Normal)
- 2300 samples, perfectly balanced
- Features: RMS, Kurtosis, Crest Factor, Skewness etc.

## Results
| Model         | Accuracy |
|---------------|----------|
| Random Forest | 97%      |
| CNN (coming)  | TBD      |

## Key Finding
Model achieve high accuracy with normal vs faulty bearings. Confusion occur only in small damaging fauls (0.007 inch) across different fault type 

## Next Steps
CNN on raw vibration signals to capture characteristic defect frequencies and improve small fault discrimination.
