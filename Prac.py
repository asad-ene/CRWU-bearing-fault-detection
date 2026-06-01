import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis


df = pd.read_csv('D:/Python and ML/CRWU bearing/test.csv')
df['peak_to_peak'] = df['max'] - df['min']
df['crest_factor'] = df['max'] / (df['rms'] + 1e-6)
df['shape_factor'] = df['rms'] / (df['mean'].abs() + 1e-6)
df['impulse_factor'] = df['max'] / (df['mean'].abs() + 1e-6)
df['margin_factor'] = df['max'] / (df['sd'] + 1e-6)
df['skewness_kurtosis'] = df['skewness'] * df['kurtosis']
df['rms_sd_ratio'] = df['rms'] / (df['sd'] + 1e-6)

X = df.drop('fault', axis=1)
Y = df['fault']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
