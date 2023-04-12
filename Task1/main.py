from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# загрузка данных
data = pd.read_csv('data.csv', sep=';')

# Условные вероятности для игры или не игры
# for i in data.columns[:-1]:
#     print((data.groupby(['Игра', i]).agg({i:'count'})/len(data)))
# разделение на признаки и метки классов
X = data.drop("Игра", axis=1)
y = data["Игра"]

# разбиение на обучающую и тестовую выборки (например, 70% и 30% соответственно)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# обучение наивного байесовского классификатора
clf = GaussianNB()
clf.fit(X_train, y_train)

# предсказание меток на тестовой выборке
y_pred = clf.predict(X_test)

# расчет метрик
acc = accuracy_score(y_test, y_pred) #Точность
prec = precision_score(y_test, y_pred, average="binary") #Точность положительного класса
rec = recall_score(y_test, y_pred, average="binary") #Полнота
f1 = f1_score(y_test, y_pred, average="binary") #F1-мера

print("\n\nМетрики:")
print("Точность:", acc)
print("Точность положительного класса:", prec)
print("Полнота:", rec)
print("F1-мера:", f1, "\n\n")

# создадим модель
model = BayesianNetwork([('Игра', 'Прогноз'), ('Игра', 'Температура'), ('Игра', 'Влажность'), ('Игра', 'Ветер'),('Прогноз', 'Температура'), ('Прогноз', 'Влажность'), ('Температура', 'Ветер')])

# вывод условных вероятностей BayesianEstimator
b_est = BayesianEstimator(model, data)
print('\nВывод условных вероятностей BayesianEstimator')
for i in data:
    print(b_est.estimate_cpd(i))

# вывод вероятности всех вариантов Прогноза при Игрa=0
w_cpd_b = b_est.estimate_cpd('Прогноз')
w_cpd_b.reduce([('Игра', 0)])
print('\nВывод вероятности всех вариантов Прогноза при Игрa=0')
print(w_cpd_b)

# вывод вероятности всех вариантов Температуры при Игрa=1 и Прогноз=2
t_cpd_b = b_est.estimate_cpd('Температура')
t_cpd_b.reduce([('Игра', 1), ('Прогноз', 2)])
print('\nВывод вероятности всех вариантов Температуры при Игрa=1 и Прогноз=2')
print(t_cpd_b)

# Вывод вероятности Влажность=1, если Игра=0, Прогноз=0
h_cpd_b = b_est.estimate_cpd('Влажность')
h_cpd_b.reduce([('Игра', 0), ('Прогноз', 0)])
print('\nВывод вероятности Влажность=1, если Игра=0, Прогноз=0')
print(h_cpd_b.values[1])

# вывод условных вероятностей MaximumLikelihoodEstimator
l_est = MaximumLikelihoodEstimator(model, data)
print('\nВывод условных вероятностей MaximumLikelihoodEstimator')
for i in data:
    print(l_est.estimate_cpd(i))

# вывод вероятности всех вариантов Прогноза при Игрa=0
w_cpd_l = l_est.estimate_cpd('Прогноз')
w_cpd_l.reduce([('Игра', 0)])
print('\nВывод вероятности всех вариантов Прогноза при Игрa=0')
print(w_cpd_l)

# вывод вероятности всех вариантов Температуры при Игрa=1 и Прогноз=2
t_cpd_l = l_est.estimate_cpd('Температура')
t_cpd_l.reduce([('Игра', 1), ('Прогноз', 2)])
print('\nВывод вероятности всех вариантов Температуры при Игрa=1 и Прогноз=2')
print(t_cpd_l)

# Вывод вероятности Влажность=1, если Игра=0, Прогноз=0
h_cpd_l = l_est.estimate_cpd('Влажность')
h_cpd_l.reduce([('Игра', 0), ('Прогноз', 0)])
print('\nВывод вероятности Влажность=1, если Игра=0, Прогноз=0')
print(h_cpd_l.values[1])


