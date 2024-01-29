import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

SEED = 28
np.random.seed(SEED)  

uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"
raw_data = pd.read_csv(uri)

worst_case_treated_data = raw_data.drop(columns=["Unnamed: 0"], axis=1)

worst_case_treated_data["modelo"] = worst_case_treated_data.idade_do_modelo + np.random.randint(-2, 3, size=10000)
worst_case_treated_data.modelo = worst_case_treated_data.modelo + abs(worst_case_treated_data.modelo.min()) + 1
worst_case_treated_data.sort_values('vendido', ascending=True, inplace=True)

X = worst_case_treated_data[["preco", "idade_do_modelo", "km_por_ano"]]
y = worst_case_treated_data["vendido"]

pipeline = ImbPipeline([
    ('Scaler', StandardScaler()),
    ('SMOTE', SMOTE(random_state=SEED)),
    ('Random Forest', RandomForestClassifier(random_state=SEED)),
])

hyper_parameters = {
    'Random Forest__max_depth': np.random.randint(3, 20, 10),
    'Random Forest__min_samples_split': np.random.randint(240, 400, 10),
    'Random Forest__min_samples_leaf': np.random.randint(100, 200, 10),
    'Random Forest__criterion': ['gini', 'entropy'],
    'Random Forest__bootstrap' : [True, False],
    'Random Forest__n_estimators': np.random.randint(110, 130, 10),
}

search = RandomizedSearchCV(pipeline, hyper_parameters, n_iter=64, cv=GroupKFold(n_splits=5), random_state=SEED)
search.fit(X, y, groups=worst_case_treated_data.modelo)
results = pd.DataFrame(search.cv_results_)
best_estimator = search.best_estimator_
print(f'Melhores Parâmetros encontrados: {search.best_params_}')

scores = cross_val_score(best_estimator, X, y, cv=GroupKFold(n_splits=5), groups=worst_case_treated_data.modelo)

mean = scores.mean() * 100
std = scores.std() * 100
print(f"Accuracy Médio: {mean}")
print(f"Intervalo: {(mean - 2 * std, mean + 2 * std)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=SEED)

dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
dummy.fit(X_train, y_train)
y_dummy_predict = dummy.predict(X_test)

print("Dummy: ", accuracy_score(y_test, y_dummy_predict))