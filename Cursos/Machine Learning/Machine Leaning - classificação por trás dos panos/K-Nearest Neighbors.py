from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd

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

model = KNeighborsClassifier(metric="euclidean")
model.fit(train_x, train_y)

algorithm_predictions = model.predict(test_x)
algorithm_accuracy = accuracy_score(test_y, algorithm_predictions)

dummy = DummyClassifier()
dummy.fit(train_x, train_y)

dummy_score = dummy.score(test_x, test_y)

print("Maria: ", model.predict(x_mary_normalized))

print(f"Acurácia do algoritmo: {algorithm_accuracy}")
print(f"Acurácia da baseline: {dummy_score}")


