import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

def train_model():
    # Load dataset
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['cultivar'] = wine.target

    # Feature selection (6 features selected)
    features = ['alcohol', 'malic_acid', 'magnesium', 'total_phenols', 'flavanoids', 'color_intensity']
    X = df[features]
    y = df['cultivar']

    # Model Pipeline with mandatory scaling for Logistic Regression
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(multi_class='multinomial', max_iter=1000))
    ])

    # Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    print("Classification Report:\n", classification_report(y_test, model.predict(X_test)))

    # Save Model
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/wine_cultivar_model.pkl')
    print("Model saved successfully.")

# Main Guard for reproducibility and robustness
if __name__ == "__main__":
    train_model()