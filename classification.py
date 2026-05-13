import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# 1. DATA GENERATION (Synthetic Dataset)

np.random.seed(42)
data_size = 1000

# Generating independent features
pages_viewed = np.random.randint(1, 25, data_size)
time_on_site = np.random.uniform(0.5, 40.0, data_size)
prev_visits = np.random.randint(0, 15, data_size)
is_ad_traffic = np.random.choice([0, 1], size=data_size)

# Creating the DataFrame
df = pd.DataFrame({
    'pages_viewed': pages_viewed,
    'time_on_site': time_on_site,
    'prev_visits': prev_visits,
    'is_ad_traffic': is_ad_traffic
})

# Mathematical Logic with Stochastic Noise
# z calculation (Logit Score)
logit_score = (df['pages_viewed'] * 0.4) + (df['time_on_site'] * 0.1) + (df['is_ad_traffic'] * 2) - 10
# Adding Gaussian Noise 
noise = np.random.normal(0, 2, data_size)
# Final Purchase decision (Sigmoid Threshold)
df['purchase'] = (logit_score + noise > 0).astype(int)


# 2. PREPROCESSING

X = df.drop('purchase', axis=1)
y = df['purchase']

# Data Splitting (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 3. TRAINING (Random Forest)
# Initializing ensemble model with 100 trees
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


# 4. EVALUATION & VISUALIZATION

y_pred = model.predict(X_test_scaled)

# Performance Metrics
print("--- Model Performance ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted: No', 'Predicted: Yes'],
            yticklabels=['Actual: No', 'Actual: Yes'])
plt.title('Confusion Matrix for Customer Purchase Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# Feature Importance 
importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance in Random Forest Model')
plt.show()
