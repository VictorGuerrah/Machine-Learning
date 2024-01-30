import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GroupKFold, cross_validate
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 28
np.random.seed(SEED)

uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"
raw_data = pd.read_csv(uri)

treated_data = raw_data.drop(columns=["Unnamed: 0"], axis=1)
treated_data["modelo"] = treated_data.idade_do_modelo + np.random.randint(-2, 3, size=10000)
treated_data.modelo = treated_data.modelo + abs(treated_data.modelo.min()) + 1

X = treated_data[["preco", "idade_do_modelo", "km_por_ano"]]
y = treated_data["vendido"]

pipeline = ImbPipeline([
    ('Scaler', StandardScaler()),
    ('SMOTE', SMOTE()),
    ('SVC', SVC())
])

cv = GroupKFold(n_splits=5)
results = cross_validate(pipeline, X, y, cv=cv, groups=treated_data.modelo, return_train_score=False)
mean_score = np.mean(results['test_score'])
std_dev = np.std(results['test_score'])

print("Resultados da Validação Cruzada:")
print("Média dos Scores de Teste: {:.3f}".format(mean_score))
print("Desvio Padrão dos Scores de Teste: {:.3f}".format(std_dev))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

pipeline.fit(X_train, y_train)
y_predict = pipeline.predict(X_test)

dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
dummy.fit(X_train, y_train)
y_dummy_predict = dummy.predict(X_test)

print("SVC: ", accuracy_score(y_test, y_predict))
print("Dummy: ", accuracy_score(y_test, y_dummy_predict))
