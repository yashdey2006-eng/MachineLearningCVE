import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
print("Loading datasets...")

df1 = pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv", low_memory=True)
df2 = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv", low_memory=True)
df3 = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", low_memory=True)

dataset = pd.concat([df1, df2, df3], ignore_index=True)
dataset.columns = dataset.columns.str.strip()
print("Datasets combined")

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

print("Converting to numeric...")
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X.replace([np.inf, -np.inf], np.nan, inplace=True)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)
print("Imputation completed")

from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# ---------------------------------------------------------
# Train/Test Split
# ---------------------------------------------------------
from sklearn.model_selection import train_test_split

# Change test_size here: 0.2, 0.4, 0.6 etc.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
print("Train-test split completed")

# ---------------------------------------------------------
# PCA
# ---------------------------------------------------------
from sklearn.decomposition import PCA

print("Starting PCA...")
pca = PCA(n_components=5)   # adjust components if needed
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print("PCA completed")

# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier

print("Starting Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,   # number of trees
    random_state=0,
    n_jobs=-1           # use all CPU cores
)
rf_model.fit(X_train, y_train)
print("Random Forest training completed")

y_pred = rf_model.predict(X_test)
print("Prediction completed")

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("\nEvaluation Results (Random Forest)")
print("="*50)
print("Accuracy   : {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision  : {:.4f}".format(precision_score(y_test, y_pred, average='weighted', zero_division=0)))
print("Recall     : {:.4f}".format(recall_score(y_test, y_pred, average='weighted', zero_division=0)))
print("F1 Score   : {:.4f}".format(f1_score(y_test, y_pred, average='weighted', zero_division=0)))
print("="*50)

print("\nClassification Report")
print("="*50)
print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("Class","Precision","Recall","F1-Score","Support"))
print("-"*50)

report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

# Per-class metrics
for i in range(len(labelencoder_y.classes_)):
    class_name = str(i)
    print("{:<8} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
        class_name,
        report[class_name]["precision"],
        report[class_name]["recall"],
        report[class_name]["f1-score"],
        int(report[class_name]["support"])
    ))

print("-"*50)
# Accuracy row
print("{:<8} {:<10} {:<10} {:<10} {:<10}".format(
    "Accuracy","","","{:.2f}".format(report["accuracy"]),int(report["weighted avg"]["support"])
))
# Macro Avg row
print("{:<8} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
    "Macro",report["macro avg"]["precision"],report["macro avg"]["recall"],report["macro avg"]["f1-score"],int(report["macro avg"]["support"])
))
# Weighted Avg row
print("{:<8} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}".format(
    "Weighted",report["weighted avg"]["precision"],report["weighted avg"]["recall"],report["weighted avg"]["f1-score"],int(report["weighted avg"]["support"])
))
print("="*50)
