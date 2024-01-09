import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("Message Group - Product.csv")

df.info()

df.describe()

missing_values_count = df.isnull().sum()
print("Missing values counted by columns:")
print(missing_values_count)

missing_values = (df.isna().sum() / df.shape[0]) * 100
missing_percentage = missing_values.round(2).sort_values(ascending=False)
print(missing_percentage)

df = df.dropna(subset=['MRP'])

if df['Discount'].dtype == 'object':
    df['Discount'] = df['Discount'].str.replace('% off', '').astype(float) / 100
else:
    df['Discount'] = df['Discount'] / 100

dfcleaned_info = df.info()
first_five_rows_cleaned = df.head()

brand_sell_prices = df.groupby('BrandName')['SellPrice'].mean()

plt.figure(figsize=(20, 6))
brand_sell_prices.plot(kind='bar', color='Blue')
plt.title('Average Sell Price by Brand')
plt.xlabel('Brand')
plt.ylabel('Average Sell Price')
plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
plt.show()
plt.savefig('Brand vs Avg_sell_price')

descriptive_stats = df[['SellPrice', 'Discount', 'MRP']].describe()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(df['SellPrice'])
plt.title('Descriptive analysis of SellPrice')

plt.subplot(1, 3, 2)
plt.hist(df['Discount'])
plt.title('Descriptive analysis of Discount')

plt.subplot(1, 3, 3)
plt.hist(df['MRP'])
plt.title('Descriptive analysis of MRP')

plt.tight_layout()
plt.show()

plt.savefig('Histograms')

df['MRP'] = pd.to_numeric(df['MRP'], errors='coerce')
df['SellPrice'] = pd.to_numeric(df['SellPrice'], errors='coerce')
df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce')
correlation = df['MRP'].corr(df['SellPrice'])
correlation1 = df['MRP'].corr(df['Discount'])
print("Correlation Coefficient between MRP and SellPrice:", correlation)
print("Correlation Coefficient between MRP and Discount:", correlation1)

plt.figure(figsize=(15, 5))


plt.subplot(1, 2, 1)
plt.scatter(df['MRP'], df['SellPrice'])
plt.title('Relationship between MRP and SellPrice')


plt.subplot(1, 2, 2)
plt.scatter(df['MRP'], df['Discount'])
plt.title('Relationship between MRP and Discount')


plt.show()

plt.savefig('Relationships')

data_cleaned = df.dropna()

data_cleaned_count = data_cleaned.shape[0]

X_cleaned = data_cleaned[['MRP']]
y_sellprice_cleaned = data_cleaned['SellPrice']
y_discount_cleaned = data_cleaned['Discount']

X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(X_cleaned, y_sellprice_cleaned, test_size=0.2,
                                                                random_state=42)
X_train_dc, X_test_dc, y_train_dc, y_test_dc = train_test_split(X_cleaned, y_discount_cleaned, test_size=0.2,
                                                                random_state=42)


model_sellprice = LinearRegression()
model_sellprice.fit(X_train_sp, y_train_sp)
predictions_sp = model_sellprice.predict(X_test_sp)


mse_sp = mean_squared_error(y_test_sp, predictions_sp)
r2_sp = r2_score(y_test_sp, predictions_sp)

print("Mean Squared Error (MSE) for Sell Price:", mse_sp)
print("R² Score for Sell Price:", r2_sp)


model_discount = LinearRegression()
model_discount.fit(X_train_dc, y_train_dc)
predictions_dc = model_discount.predict(X_test_dc)


mse_dc = mean_squared_error(y_test_dc, predictions_dc)
r2_dc = r2_score(y_test_dc, predictions_dc)

print("Mean Squared Error (MSE) for Discount:", mse_dc)
print("R² Score for Discount:", r2_dc)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x=X_test_sp['MRP'], y=y_test_sp)
plt.plot(X_test_sp['MRP'], predictions_sp, color='red')
plt.title('Linear Regression for SellPrice')
plt.xlabel('MRP')
plt.ylabel('SellPrice')
plt.show()
