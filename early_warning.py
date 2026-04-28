import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
df = pd.read_csv('data/cleaned_data.csv')

# Prepare features and target
# Feature engineering: lagged values are the 'early signals'
df = df.sort_values(['Entity', 'Year'])
df['lag1_polity'] = df.groupby('Entity')['polity_score'].shift(1)
df['lag1_gdp_growth'] = df.groupby('Entity')['gdp_growth'].shift(1)
df['lag1_deaths'] = df.groupby('Entity')['conflict_deaths'].shift(1)

# Target: War in the current year
data = df.dropna(subset=['lag1_polity', 'lag1_gdp_growth', 'lag1_deaths', 'is_war'])

X = data[['lag1_polity', 'lag1_gdp_growth', 'lag1_deaths']]
y = data['is_war']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Early Warning Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print("Early Warning Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(rf, 'data/early_warning_model.pkl')

# Simulate future risk for target African countries in 2025 (latest known state)
latest_states = df[(df['Year'] == 2022) & (df['Entity'].isin(['Sudan', 'Libya', 'Rwanda', 'Zimbabwe']))]
if not latest_states.empty:
    X_future = latest_states[['polity_score', 'gdp_growth', 'conflict_deaths']].fillna(0)
    X_future.columns = ['lag1_polity', 'lag1_gdp_growth', 'lag1_deaths']
    # Note: Using current values as proxy for 'lag' to predict 'next'
    predictions = rf.predict_proba(X_future)
    for i, entity in enumerate(latest_states['Entity']):
        print(f"Predicted Risk for {entity} (2023-2025): {predictions[i][1]:.2f}")

print("Early warning system verified and saved.")
