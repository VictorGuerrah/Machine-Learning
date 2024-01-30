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

uri = "C:/Users/victo/OneDrive/Área de Trabalho/Victor/Bathtub/Repository/Projeto/Dataset.csv"

try:
    raw_data = pd.read_csv(uri)
    raw_data['origin / session_media'] = raw_data['origin / session_media'].astype(str)

    treated_data = raw_data.copy()
    treated_data['origin'] = treated_data['origin / session_media'].str.split(' / ').str[0]
    treated_data['session_media'] = treated_data['origin / session_media'].str.split(' / ').str[1]
    treated_data.drop('origin / session_media', axis=1, inplace=True)

except Exception as e:
    print(f"Erro: {e}")

print(treated_data.head())
print(raw_data.head())



# X = treated_data[["preco", "idade_do_modelo", "km_por_ano"]]
# y = treated_data["vendido"]

# pipeline = ImbPipeline([
#     ('Scaler', StandardScaler()),
#     ('SMOTE', SMOTE(random_state=SEED)),
#     ('Random Forest', RandomForestClassifier(random_state=SEED)),
# ])

# hyper_parameters = {
#     'Random Forest__max_depth': np.random.randint(3, 20, 10),
#     'Random Forest__min_samples_split': np.random.randint(240, 400, 10),
#     'Random Forest__min_samples_leaf': np.random.randint(100, 200, 10),
#     'Random Forest__criterion': ['gini', 'entropy'],
#     'Random Forest__bootstrap' : [True, False],
#     'Random Forest__n_estimators': np.random.randint(110, 130, 10),
# }

# search = RandomizedSearchCV(pipeline, hyper_parameters, n_iter=64, cv=GroupKFold(n_splits=5), random_state=SEED)
# search.fit(X, y, groups=treated_data.modelo)
# results = pd.DataFrame(search.cv_results_)
# best_estimator = search.best_estimator_
# print(f'Melhores Parâmetros encontrados: {search.best_params_}')

# scores = cross_val_score(best_estimator, X, y, cv=GroupKFold(n_splits=5), groups=treated_data.modelo)

# mean = scores.mean() * 100
# std = scores.std() * 100
# print(f"Accuracy Médio: {mean}")
# print(f"Intervalo: {(mean - 2 * std, mean + 2 * std)}")

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=SEED)

# dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
# dummy.fit(X_train, y_train)
# y_dummy_predict = dummy.predict(X_test)

# print("Dummy: ", accuracy_score(y_test, y_dummy_predict))