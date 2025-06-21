import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("C:/Users/madhu/Downloads/archive (1)/Housing.csv")
df = pd.DataFrame(data)
print(df)
X = df[['sqft_living', 'bathrooms', 'bedrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

new_house = np.array([[2500, 3, 4]])  
predicted_price = model.predict(new_house)
print(f"Predicted price for house: ${predicted_price[0]:,.2f}")
