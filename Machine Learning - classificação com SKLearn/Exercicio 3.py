from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

SEED = 20
np.random.seed(SEED)

uri = 'https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv'
data = pd.read_csv(uri)

treatment = {0 : 1, 1: 0}
data["finished"] = data.unfinished.map(treatment)

x = data[["expected_hours", "price"]]
y = data["finished"]

raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, random_state=SEED, test_size=0.2, stratify=y)

scaler = StandardScaler()
scaler.fit(raw_train_x)
train_x = scaler.transform(raw_train_x)
test_x = scaler.transform(raw_test_x)

model = SVC()
model.fit(train_x, train_y)

predictions = model.predict(test_x)

algorithm_accuracy = accuracy_score(test_y, predictions)
baseline_accuracy = accuracy_score(test_y, np.ones(len(test_y)))

print(f"Acurácia do algoritmo: {algorithm_accuracy}")
print(f"Acurácia da baseline: {baseline_accuracy}")

sns.scatterplot(x="expected_hours", y="price", hue="finished", data=data)
plt.show()
