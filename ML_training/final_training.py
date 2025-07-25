import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib
#load a csv and limit number of samples per class
def load_limited_per_person(csv_path, max_per_person=512):
    df = pd.read_csv(csv_path)
    dfs = []
    for label in df['label'].unique():
        df_label = df[df['label'] == label].head(max_per_person)
        dfs.append(df_label)
    limited_df = pd.concat(dfs).reset_index(drop=True)

    X = limited_df.drop(columns=["image_path", "label"]).values
    y = limited_df["label"].values
    return np.array(X), np.array(y)
# split train and test data
def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_idx = int(len(X) * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def standard_scaler(X, mean=None, std=None):
    X = np.array(X, dtype=float)
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        std[std == 0] = 1e-8
    return (X - mean) / std, mean, std

def manual_accuracy_score(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))

# load old data
print("Loading old data from hog_features.csv...")
X_old, y_old = load_limited_per_person("final_features.csv", max_per_person=512)
print(f"Old data shape: {X_old.shape}, Labels: {y_old.shape}")

#load aug
print("Loading new data from aug_features.csv...")
X_new, y_new = load_limited_per_person("aug_features.csv", max_per_person=2300)
print(f"New data shape: {X_new.shape}, Labels: {y_new.shape}")

# combine data
X_combined = np.concatenate((X_old, X_new), axis=0)
y_combined = np.concatenate((y_old, y_new), axis=0)

print(f"Combined data shape: {X_combined.shape}, Labels: {y_combined.shape}")

if X_combined.shape[0] < 10:
    print("[ERROR] Not enough data to train. Exiting...")
    exit()

# Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Scale
X_train_scaled, mean, std = standard_scaler(X_train)
X_test_scaled, _, _ = standard_scaler(X_test, mean, std)

# Train
print("\nTraining Decision Tree on combined dataset...")
dt_model = DecisionTreeClassifier(random_state=42, criterion='entropy',max_depth=13)
dt_model.fit(X_train_scaled, y_train)

# Predict + Evaluate
y_pred = dt_model.predict(X_test_scaled)
accuracy = manual_accuracy_score(y_test, y_pred)

print(f"\nAccuracy on test set: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model 
joblib.dump({
    'model': dt_model,
    'mean': mean,
    'std': std
}, 'decision_tree_hog_model_combined.pkl')

print("\nModel saved as 'decision_tree_hog_model_combined.pkl'")



# Predict + Evaluate (on training set)
y_train_pred = dt_model.predict(X_train_scaled)
train_accuracy = manual_accuracy_score(y_train, y_train_pred)
print(f"Training set accuracy: {train_accuracy * 100:.2f}%")

# Already present: test accuracy
print(f"Test set accuracy: {accuracy * 100:.2f}%")
