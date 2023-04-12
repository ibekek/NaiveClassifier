import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import TreeSearch, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork

# Загрузка датасета
adult_data = pd.read_csv('adult.csv')

target = 'income'

# Создание модели
search = TreeSearch(adult_data, root_node=target)
best_model = search.estimate()

# Обучение модели
model = BayesianNetwork(best_model.edges())
model.fit(adult_data, estimator=MaximumLikelihoodEstimator)

# Отрисовка графа
pos = nx.circular_layout(model)
nx.draw(model, pos, with_labels=True)
plt.show()

# Разделение выборок на train и test
train_data = adult_data[:39074]
test_data = adult_data[39074:]

# Копия test data без учета target
test_data_copy = test_data.drop(target, axis=1)

# Предсказания и расчет точности
y_pred = model.predict(test_data_copy).values
y_true = test_data[target].values
accuracy = (y_pred == y_true).mean()
print('Accuracy: {:.4f}'.format(accuracy))
