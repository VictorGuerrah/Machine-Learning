from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SEED = 20
np.random.seed(SEED)

uri = 'https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv'
data = pd.read_csv(uri)

treatment = {"yes" : 1, "no": 0}
data.sold = data.sold.map(treatment)
data["model_age"] = datetime.today().year - data.model_year
data["kilometer_per_year"] = 1.60934 * data.mileage_per_year
data = data.drop(columns=["Unnamed: 0", "mileage_per_year", "model_year"])

x = data[["kilometer_per_year", "model_age", "price"]]
y = data["sold"]

raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, random_state=SEED, test_size=0.2, stratify=y)

scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

model = DecisionTreeClassifier(max_depth=2)
model.fit(raw_train_x, train_y)

algorithm_predictions = model.predict(raw_test_x)
algorithm_accuracy = accuracy_score(test_y, algorithm_predictions)

dummy = DummyClassifier()
dummy.fit(raw_train_x, train_y)

dummy_score = dummy.score(raw_test_x, test_y)

print(f"Acurácia do algoritmo: {algorithm_accuracy}")
print(f"Acurácia da baseline: {dummy_score}")

plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=x.columns, class_names=["Não Vendido", "Vendido"], rounded=True)
plt.show()
