# Apriori

# Importing the libraries
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
#min_support in this example = 3*7/7500 ,i.e. considering a customer purchased 3 items in a day, then a week's purchase(considering the tranactions are recorded for a week) would be 3*7, then divided by the total transactions
#min_length = minimun no. of products to be allowed in the rule, here, association b/w 2 diff. products
#min_confidence = all the rules obtained will be true atleast 20% of the time
#min_lift = relevant rules
# Visualising the results
results = list(rules)
print(results)