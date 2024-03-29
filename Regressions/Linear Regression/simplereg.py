# Simple Linear Regression

# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1]

# A faceted simple linear regression
g = sns.lmplot(x='YearsExperience',y='Salary',data=dataset,truncate=False,y_jitter=.02)
g.set(xlim=(0,11),ylim=(30000,130000))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
finalpred = pd.DataFrame(data=y_pred,columns=['Prediction'],index=y_test)
print(f"The final prediction of Salary using Simple Linear Regression are as follows: \n {finalpred}")

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
