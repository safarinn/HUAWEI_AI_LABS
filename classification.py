import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# 1. Create Synthetic Dataset
np.random.seed(42)
data_size = 1000

data = {
    'pages_viewed': np.random.randint(1, 25, data_size),
    'time_on_site': np.random.uniform(0.5, 40.0, data_size),
    'prev_visits': np.random.randint(0, 15, data_size),
    'is_ad_traffic': np.random.choice([0, 1], data_size)
}

df = pd.DataFrame(data)

# Logic for "Purchase": More likely if they view many pages or spend much time
# We add a little noise (randomness) so the model isn't 100% perfect
logit = (df['pages_viewed'] * 0.4 + df['time_on_site'] * 0.1 + (df['is_ad_traffic'] * 2) - 10)
probabilities = 1 / (1 + np.exp(-logit))
df['purchase'] = (probabilities > 0.5).astype(int)

# 2. Prepare Data
X = df.drop('purchase', axis=1)
y = df['purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Generate Predictions & Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# 5. Visualize (The "Deep Dive" Plot)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted: No', 'Predicted: Yes'],
            yticklabels=['Actual: No', 'Actual: Yes'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for Customer Purchase Prediction')
plt.show()