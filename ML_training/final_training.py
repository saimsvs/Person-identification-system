import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report  
import joblib  # Used to save the trained model

# load a CSV and limit number of samples per person/class
def load_limited_per_person(csv_path, max_per_person=512):
    df = pd.read_csv(csv_path)  # Read the CSV file into a DataFrame
    dfs = []
    for label in df['label'].unique():
        df_label = df[df['label'] == label].head(max_per_person)
        dfs.append(df_label)
    limited_df = pd.concat(dfs).reset_index(drop=True)

    # Extract features 
    X = limited_df.drop(columns=["image_path", "label"]).values
    y = limited_df["label"].values
    return np.array(X), np.array(y)

# Custom train-test split function 
def train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Compute index to split the data
    split_idx = int(len(X) * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Standardize the feature vectors (mean = 0, std = 1)
def standard_scaler(X, mean=None, std=None):
    X = np.array(X, dtype=float)
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        std[std == 0] = 1e-8  # Prevent division by zero
    return (X - mean) / std, mean, std

# Custom function to compute accuracy
def manual_accuracy_score(y_true, y_pred):
    return np.mean(np.array(y_true) == np.array(y_pred))


# load original HOG-based feature data (pre-augmentation)
print("Loading old data from hog_features.csv...")
X_old, y_old = load_limited_per_person("final_features.csv", max_per_person=512)
print(f"Old data shape: {X_old.shape}, Labels: {y_old.shape}")

# load augmented feature data
print("Loading new data from aug_features.csv...")
X_new, y_new = load_limited_per_person("aug_features.csv", max_per_person=2000)
print(f"New data shape: {X_new.shape}, Labels: {y_new.shape}")

# combine original and augmented datasets
X_combined = np.concatenate((X_old, X_new), axis=0)
y_combined = np.concatenate((y_old, y_new), axis=0)
print(f"Combined data shape: {X_combined.shape}, Labels: {y_combined.shape}")

# safety check: Make sure we have enough data to proceed
if X_combined.shape[0] < 10:
    print("[ERROR] Not enough data to train. Exiting...")
    exit()

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# standardize features using training set statistics
X_train_scaled, mean, std = standard_scaler(X_train)
X_test_scaled, _, _ = standard_scaler(X_test, mean, std)  # Apply same mean/std to test set

# train a Decision Tree classifier
print("\nTraining Decision Tree on combined dataset...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=23)  # Set max depth to prevent overfitting
dt_model.fit(X_train_scaled, y_train)

# predict on test data and evaluate performance
y_pred = dt_model.predict(X_test_scaled)
accuracy = manual_accuracy_score(y_test, y_pred)

# results
print(f"\nAccuracy on test set: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))  # Includes precision, recall, F1-score, support, etc.

# save the trained model and normalization parameters 
joblib.dump({
    'model': dt_model,
    'mean': mean,
    'std': std
}, 'decision_tree_hog_model_combined.pkl')

print("\nModel saved as 'decision_tree_hog_model_combined.pkl'")