import pandas as pd
from matplotlib import pyplot as plt, MatplotlibDeprecationWarning
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
from sklearn.tree import plot_tree

data = pd.read_csv('Message Group - Product.csv')
correlation_matrix = data.corr()
data['MRP'] = data['MRP'].replace('#REF!',0)
data['MRP'] = data['MRP'].replace('',0)


data['break'] = data['Discount'].str.split('%')
data[['Discount_percentage', 'Discount_type']] = data['break'].apply(pd.Series)
data['Discount'] = pd.to_numeric(data['Discount_percentage'],errors='coerce')
data['SellPrice'] = pd.to_numeric(data['SellPrice'],errors='coerce')
data['MRP'] = data['MRP'].replace('#REF!',0)
data['MRP'] = data['MRP'].replace('',0)
data['MRP'] = pd.to_numeric(data['MRP'],errors='coerce')





data = data.drop(['Product Size','S.No', 'Product ID', 'Brand Desc','Currancy','break','Discount_type','Product Name','Discount_percentage'], axis=1)
data["MRP"].fillna(method="ffill", inplace=True)
data.info()
print(data.isnull().sum())
le = LabelEncoder()
data['Category'] = le.fit_transform(data['Category'])
data['BrandName'] = le.fit_transform(data['BrandName'])
#data = pd.get_dummies(data, columns=['Category', 'BrandName'])
#data['Product Size'] = le.fit_transform(data['Product Size'])
X = data.drop('SellPrice', axis=1)
y = data['SellPrice']

'''kf = KFold(n_splits=5, shuffle=True, random_state=42)
#Find optimal number of trees
n_estimators = [50, 100, 150, 200, 250, 300, 350]
for val in n_estimators:
    score = cross_val_score(ensemble.RandomForestRegressor(n_estimators= val, random_state= 42), X, y, cv = kf, scoring="neg_root_mean_squared_error")
    print(f'Average score({val}): {"{:.3f}".format(score.mean())}')

n_depth = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
for val in n_depth:
    score = cross_val_score(ensemble.RandomForestRegressor(n_estimators= 100, max_depth = val, random_state= 42), X, y, cv = kf, scoring="neg_root_mean_squared_error")
    print(f'Average score({val}): {"{:.3f}".format(score.mean())}')'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=50, random_state=42,max_depth=14)
rf_model.fit(X_train, y_train)


warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)



predictions_train = rf_model.predict(X_train)
predictions_test = rf_model.predict(X_test)


mse_train = mean_squared_error(y_train, predictions_train)
mse_test = mean_squared_error(y_test, predictions_test)

mean_target = y.mean()
percentage_error_train = (mse_train / mean_target) * 100
percentage_error_test = (mse_test / mean_target) * 100

print(rf_model.feature_importances_)
print(f'Mean Squared Error (Training): {mse_train}')
print(f'Mean Squared Error (Test): {mse_test}')
print(f'Percentage Error (Training): {percentage_error_train:.2f}%')
print(f'Percentage Error (Test): {percentage_error_test:.2f}%')


plt.figure(figsize=(20, 10))
plot_tree(rf_model.estimators_[0], feature_names=['BrandName', 'MRP', 'Discount', 'Category','Product Size'], filled=True, rounded=True)
plt.title("Decision Tree in Random Forest")
plt.show()
correlation_matrix = data.corr()



threshold = 0.2


accuracy_train = (abs(predictions_train - y_train) / y_train) <= threshold
accuracy_test = (abs(predictions_test - y_test) / y_test) <= threshold


accuracy_train_percentage = (accuracy_train.sum() / len(y_train)) * 100
accuracy_test_percentage = (accuracy_test.sum() / len(y_test)) * 100

print(f'Accuracy (Training): {accuracy_train_percentage:.2f}%')
print(f'Accuracy (Test): {accuracy_test_percentage:.2f}%')

