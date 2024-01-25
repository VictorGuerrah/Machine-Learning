from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SEED = 20
THRESHOLD = 0.99
np.random.seed(SEED)

uri = 'C:/Users/victo/OneDrive/Área de Trabalho/Victor/Bathtub/Repository/Machine Learning - lidando com dados de muitas dimensões/exames.csv'
raw_data = pd.read_csv(uri)

X = raw_data.drop(columns=["id", "diagnostico", "exame_33"])
y = raw_data.diagnostico

vars_to_remove = set()
corr_matrix = X.corr()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > THRESHOLD:
            vars_to_remove.add(corr_matrix.columns[j])

X = X.drop(columns=list(vars_to_remove))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, test_size=0.2, stratify=y)

if y_train.value_counts()[1] / y_train.value_counts()[0] < 0.3:
    smote = SMOTE()
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
else:
    X_train_smote, y_train_smote = X_train, y_train

model = RandomForestClassifier(n_estimators=100)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rfecv', RFECV(estimator=model, cv=5, scoring='accuracy', step=1))
])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100)
rfecv = RFECV(estimator=model, cv=5, scoring='accuracy', step=1)
rfecv.fit(X_train_scaled, y_train_smote)

X_rfecv_train = rfecv.transform(X_train_scaled)
X_rfecv_test = rfecv.transform(X_test_scaled)

model.fit(X_rfecv_train, y_train_smote)

algorithm_predictions = model.predict(X_rfecv_test)
algorithm_accuracy = accuracy_score(y_test, algorithm_predictions)
confusion_mtx = confusion_matrix(y_test, algorithm_predictions)
classification_rpt = classification_report(y_test, algorithm_predictions)

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_rfecv_train, y_train_smote)
dummy_score = dummy.score(X_rfecv_test, y_test)

print(f"Número de características selecionadas: {rfecv.n_features_}")
print(f"Acurácia do algoritmo: {algorithm_accuracy}")
print("Matriz de Confusão:\n", confusion_mtx)
print("Relatório de Classificação:\n", classification_rpt)
print(f"Acurácia da baseline: {dummy_score}")

tsne = TSNE(n_components = 2)
X_test_scaled = tsne.fit_transform(X_test_scaled)
plt.figure(figsize=(14,8))
sns.scatterplot(x = X_test_scaled[:,0] , y = X_test_scaled[:,1], hue = y_test)
plt.show()
