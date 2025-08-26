import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# STEP 1: Generate Synthetic Dataset
# ---------------------------
np.random.seed(42)

house_types = ['Flat', 'Bungalow', 'Duplex', 'Triplex', 'Tenament']
cities = ['Vadodara', 'Ankleshwar', 'Surat']
areas = {
    'Vadodara': ['Alkapuri', 'Gotri', 'Manjalpur', 'Akota', 'Subhanpura'],
    'Ankleshwar': ['GIDC', 'Valia Road', 'Kapodra', 'Rajpipla Road'],
    'Surat': ['Adajan', 'Vesu', 'Katargam', 'Piplod', 'Varachha']
}

city_multiplier = {
    'Vadodara': 350,
    'Ankleshwar': 250,
    'Surat': 400
}

type_multiplier = {
    'Flat': 1.0,
    'Bungalow': 1.5,
    'Duplex': 1.3,
    'Triplex': 1.7,
    'Tenament': 0.9
}

records = []
for _ in range(50):
    house_type = random.choice(house_types)
    city = random.choice(cities)
    area = random.choice(areas[city])
    size = np.random.randint(600, 5000)
    rooms = np.random.randint(1, 8)
    age = np.random.randint(0, 31)

    base_price = size * city_multiplier[city]
    price = base_price * type_multiplier[house_type]
    price += (rooms * 10000) - (age * 1500)
    price += np.random.randint(-50000, 50000)

    records.append({
        'Size': size,
        'HouseType': house_type,
        'City': city,
        'Area': area,
        'Rooms': rooms,
        'Age': age,
        'Price': int(price)
    })

df = pd.DataFrame(records)

# Save the generated dataset
df.to_csv('synthetic_house_prices.csv', index=False)
print("Dataset saved as 'synthetic_house_prices.csv'")

print("Sample Data:")
print(df.head())

# ---------------------------
# STEP 2: EDA
# ---------------------------
print("\nData Info:")
print(df.info())

print("\nData Description:")
print(df.describe())

# Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Price'], kde=True)
plt.title('Price Distribution')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ---------------------------
# STEP 3: Preprocessing
# ---------------------------
X = df.drop('Price', axis=1)
y = df['Price']

categorical_features = ['HouseType', 'City', 'Area']
numerical_features = ['Size', 'Rooms', 'Age']

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)


# ---------------------------
# STEP 4: Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# STEP 5: Model Training
# ---------------------------
# Linear Regression Pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

# Random Forest Pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

# ---------------------------
# STEP 6: Evaluation
# ---------------------------

def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}\n")

evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)

# ---------------------------
# STEP 7: Visualizations
# ---------------------------
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_lr, color='blue', label='Linear Regression')
plt.scatter(y_test, y_pred_rf, color='green', label='Random Forest')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.legend()
plt.show()
