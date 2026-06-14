import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv('apistox_data.csv')

feature_cols = ['year', 'herbicide', 'fungicide', 'insecticide', 
                'other_agrochemical', 'source', 'toxicity_type']
X = df[feature_cols]
y = df['label']

numeric_features = ['year', 'herbicide', 'fungicide', 'insecticide', 'other_agrochemical']
categorical_features = ['source', 'toxicity_type']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.3f}")
joblib.dump(model, 'agrochemical_model.pkl')
print("Model saved as 'agrochemical_model.pkl'")