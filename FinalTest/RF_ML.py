from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.impute import SimpleImputer
from joblib import dump
import pandas as pd
import numpy as np

# Load your dataset
file_path = 'modified_dataset.csv'
df = pd.read_csv(file_path)

feature_columns = [
    "nb_links"
]

# Separate the features and the target variable
X = df[feature_columns]  # Features
y = df['is_phish']

# Check if there are any missing values in the dataset
print("Missing values in the dataset:")
print(X.isnull().sum())

# Impute missing values
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X_imputed = imputer.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=feature_columns)

# Split the dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X_imputed_df, y, test_size=0.2, stratify=y, random_state=42)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training set
rf_classifier.fit(X_train, y_train)

# Evaluate the model on the training data
train_predictions = rf_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f'Training Accuracy: {train_accuracy}')

# Make predictions on the testing set
predictions = rf_classifier.predict(X_test)
prediction_proba = rf_classifier.predict_proba(X_test)

# Evaluate the model using accuracy and log loss
accuracy = accuracy_score(y_test, predictions)
log_loss_val = log_loss(y_test, prediction_proba, labels=[0, 1])

print(f'Testing Accuracy: {accuracy}')
print(f'Log Loss: {log_loss_val}')

# Output feature importance
print("Feature Importances:")
print(pd.DataFrame(rf_classifier.feature_importances_, index=X_train.columns, columns=['Importance']).sort_values('Importance', ascending=False))

# Save the trained model to a file
dump(rf_classifier, 'Final.joblib')

print("Model trained and saved successfully.")