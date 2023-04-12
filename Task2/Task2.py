import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import TreeSearch, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork

# Load the dataset
adult_data = pd.read_csv('adult.csv')

# Define the list of features and the target variable
features = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
target = 'income'

# Create the TreeSearch object and learn the model structure using TAN algorithm
search = TreeSearch(adult_data, root_node=target)
best_model = search.estimate()

# Fit the model with the data and print the model parameters
model = BayesianNetwork(best_model.edges())
model.fit(adult_data, estimator=MaximumLikelihoodEstimator)

# Draw the graph of the model
pos = nx.circular_layout(model)
nx.draw(model, pos, with_labels=True)
plt.show()

# Split the data into training and testing sets
train_data = adult_data[:39074]
test_data = adult_data[39074:]

# Make a copy of the test data and drop the target variable
test_data_copy = test_data.drop(target, axis=1)

# Predict the target variable for the test data and calculate the accuracy
y_pred = model.predict(test_data_copy).values
y_true = test_data[target].values
accuracy = (y_pred == y_true).mean()
print('Accuracy: {:.4f}'.format(accuracy))
