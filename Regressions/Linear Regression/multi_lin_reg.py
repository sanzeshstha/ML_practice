#Import Data
import pandas as pd
import seaborn as sns
sns.set()

#Load data
data = pd.read_csv("50_Startups.csv")
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

#Pre-process data
s= (X.dtypes == 'object')
object_col = list(s[s].index)
num_col = X.drop(object_col,axis=1)

from sklearn.preprocessing import OneHotEncoder
# Apply one-hot encoder to each column with categorical data
encode = OneHotEncoder(handle_unknown='ignore',sparse=False)
OH_col_X = pd.DataFrame(encode.fit_transform(X[object_col]))

OH_col_X.index = X.index

final_X = pd.concat([OH_col_X,num_col],axis=1)

#Split the data
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_test = train_test_split(final_X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)
y_pred = reg.predict(X_valid)

finalpred = pd.DataFrame(data=y_pred,columns=['Prediction'],index=y_test)
print(f"The profit as predicted by the linear regression are as follows:\n{finalpred}")

#Applying backward elimination, we find R&D spend and profit to be linearly co-related
import statsmodels.api as sm
new_X = final_X.copy()
new_X = sm.add_constant(new_X)
x1 = new_X.iloc[:,[0,4]]
result=sm.OLS(y,x1).fit()
result.summary()

#Visualization of the co-related variables
sns.lmplot(data=data,x="R&D Spend", y="Profit", hue="State",truncate=False,
           x_jitter=0.05,y_jitter=0.3)
