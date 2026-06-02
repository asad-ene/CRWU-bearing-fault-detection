# Bearing Fault Detection — CWRU Dataset

## Problem
Predict bearing fault type from vibration signal features using machine learning.

## Dataset
Case Western Reserve University (CWRU) Bearing Dataset
- 10 classes (3 fault types + Normal)
- 2300 samples, perfectly balanced
- Features: RMS, Kurtosis, Crest Factor, Skewness etc.

## Hyperparaters setting Random Forest:
- n_estimators=200,
- criterion='entropy',
- min_samples_split=5,
- max_depth=20,
- random_state=42

## Hyperparameter settings for SVM:
- kernel='rbf'
- C=10
- gamma=1

| Model | Accuracy | Training Time | Prediction Time |
|-------|----------|---------------|-----------------|
| Random Forest | **97%** | 2-3 seconds | <0.1 seconds |
| SVM (RBF) | 95% | 3-4 minutes | 0.5 seconds |


## Key Finding
Model achieve high accuracy with normal vs faulty bearings. Confusion occur only in small damaging fauls (0.007 inch) across different fault type. Random forest outperform SVM by 2 percent, SVM has slow performance because of one v one strategy. Both model achieve achieve 95 plus accuracy

## Next Step
Model training raw vibration signals to capture characteristic defect frequencies and improve small fault discrimination.
