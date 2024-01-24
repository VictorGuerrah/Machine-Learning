from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 20
np.random.seed(SEED)

uri = 'C:/Users/victo/OneDrive/Área de Trabalho/Victor/Bathtub/Repository/Machine Leaning - classificação por trás dos panos/Customer-Churn.csv'
raw_data = pd.read_csv(uri)

treatment = {"Sim" : 1, "Nao": 0}
data = raw_data[['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn']].replace(treatment)
data_aux = pd.get_dummies(raw_data.drop(['Conjuge', 'Dependentes', 'TelefoneFixo', 'PagamentoOnline', 'Churn'], axis=1))
data_aux = data_aux.astype(int)
data = pd.concat([data, data_aux], axis=1)

x = data.drop(columns="Churn")
y = data["Churn"]

smt = SMOTE()
x, y = smt.fit_resample(x, y)

scaler = StandardScaler()
x_normalized = scaler.fit_transform(x)

x_mary = [[0,0,1,1,0,0,39.90,1,0,0,0,1,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,1]]
x_mary_normalized = scaler.transform(pd.DataFrame(x_mary, columns = x.columns))

train_x, test_x, train_y, test_y = train_test_split(x_normalized, y, random_state=SEED, test_size=0.2, stratify=y)
# ------------------------

bnb = BernoulliNB(binarize=0.44)
bnb.fit(train_x, train_y)

algorithm_predictions_bnb = bnb.predict(test_x)
algorithm_accuracy = accuracy_score(test_y, algorithm_predictions_bnb)
algorithm_precision = precision_score(test_y, algorithm_predictions_bnb)
bnb_predict = bnb.predict(x_mary_normalized)

print(f"Acurácia do algoritmo BNB: {algorithm_accuracy}") # Desempenho do algoritmo de uma forma geral
print(f"Precisão do algoritmo BNB: {algorithm_precision}") # Desemprenho do algoritmo em relação as previsões Positivas Corretas. De todos os clientes que o modelo classificou como "churn", quantos são realmente churn?
print(f"Sensibilidade do algoritmo BNB: {algorithm_precision}") # Desempenho do algoritmo em relação ao real valor de Verdadeiros Positivos. De todos os clientes que são realmente churn, quantos o modelo conseguiu identificar corretamente?
 
print(confusion_matrix(test_y, algorithm_predictions_bnb))

# ------------------------

knn = KNeighborsClassifier(metric="euclidean")
knn.fit(train_x, train_y)

algorithm_predictions_knn = knn.predict(test_x)
algorithm_accuracy = accuracy_score(test_y, algorithm_predictions_knn)
algorithm_precision = precision_score(test_y, algorithm_predictions_knn)
algorithm_recall = recall_score(test_y, algorithm_predictions_knn)
knn_predict = knn.predict(x_mary_normalized)

print(f"Acurácia do algoritmo KNN: {algorithm_accuracy}")
print(f"Precisão do algoritmo KNN: {algorithm_precision}")
print(f"Sensibilidade do algoritmo KNN: {algorithm_recall}")
print(confusion_matrix(test_y, algorithm_predictions_knn))

# ------------------------

decision_tree = DecisionTreeClassifier(criterion="entropy", max_depth=2)
decision_tree.fit(train_x, train_y)

algorithm_predictions_decision_tree = decision_tree.predict(test_x)
algorithm_accuracy = accuracy_score(test_y, algorithm_predictions_decision_tree)
algorithm_precision = precision_score(test_y, algorithm_predictions_decision_tree)
algorithm_recall = recall_score(test_y, algorithm_predictions_decision_tree)
decision_tree_predict = decision_tree.predict(x_mary_normalized)

print(f"Acurácia do algoritmo Decision Tree: {algorithm_accuracy}")
print(f"Precisão do algoritmo Decision Tree: {algorithm_precision}")
print(f"Sensibilidade do algoritmo Decision Tree: {algorithm_recall}")
print(confusion_matrix(test_y, algorithm_predictions_decision_tree))

dummy = DummyClassifier()
dummy.fit(train_x, train_y)
dummy_score = dummy.score(test_x, test_y)

print("Maria: ", bnb_predict, knn_predict, decision_tree_predict)
print(f"Acurácia da baseline: {dummy_score}")

