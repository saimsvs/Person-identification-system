import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import joblib  #saves/loads the trained model for future use

# Load features
# iterates over folders like from 1 to 11
#loads feauture vectors from .csv files
#x = feature matrix , y = label vector(1-11)

def load_features(base_dir="person_concat_features"):
    X, y = [], []

    for person_id in range(1, 12):
        folder = f"{base_dir}/person{person_id:02d}_concat"

        if not os.path.exists(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue

        for csv_file in os.listdir(folder):
            if csv_file.endswith(".csv"):
                file_path = os.path.join(folder, csv_file)
                features = np.loadtxt(file_path, delimiter=",")
                features = np.atleast_1d(features)
                X.append(features.flatten())
                y.append(person_id - 1)

    return np.array(X), np.array(y)

# train test split

def manual_train_test_split(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)   #seed(it gives you a same result on every run)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # decides how much of the dataset is to go for train and test

    split_idx = int(len(X) * (1 - test_size))
    train_idx = indices[:split_idx]   
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]   #return shuffled featrues(x) and corresponding labels(y)

# Standard Scaler
# to set mean = 0, SD = 1, to prevent large values from dominating the model (like basic mean cal)

def manual_standard_scaler(X, mean=None, std=None):
    X = np.array(X, dtype=float)    #ensure X is a numpy array and converted to float
    if mean is None:
        mean = np.mean(X, axis=0)     #across rows 
    if std is None:                   
        std = np.std(X, axis=0)
        std[std == 0] = 1e-8
    return (X - mean) / std, mean, std


# Accuracy Score (num of correct predictions / total num of predictions)

def manual_accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)   #actual labels
    y_pred = np.array(y_pred)  #predicted labels
    return np.mean(y_true == y_pred)   # element comparison to return a boolean value


print(" Loading features...")
X, y = load_features()  #function call
print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")  #how many samples and features were loaded 

if X.size == 0:
    print("[ERROR] No data loaded.")
    exit()

# Split the data
X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
X_train_scaled, mean, std = manual_standard_scaler(X_train)
X_test_scaled, _, _ = manual_standard_scaler(X_test, mean, std)

# Train Decision Tree
print("\n Training Decision Tree classifier...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = dt_model.predict(X_test_scaled)

accuracy = manual_accuracy_score(y_test, y_pred)
print(f"\nDecision Tree Accuracy: {accuracy * 100:.2f}%")
print("\n Classiftion Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump({
    'model': dt_model,
    'mean': mean,
    'std': std
}, 'decision_tree_person_id_model.pkl')

print("\nModel saved as 'decision_tree_person_id_model.pkl'")

# output: 
#precison, recall, f1-score, support, accuracy, macro avg, weighted avg are all metrics of scikit