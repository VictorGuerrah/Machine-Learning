from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

SEED = 20

uri = 'https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'

data = pd.read_csv(uri)

x = data[["home", "how_it_works", "contact"]]
y = data["bought"]

train_x = x[:79]
train_y = y[:79]

test_x = x[79:]
test_y = y[79:]

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=SEED , test_size= 0.2, stratify=y)

model = LinearSVC(dual=True)
model.fit(train_x, train_y)

predictions = model.predict(test_x)
accuracy = accuracy_score(test_y, predictions)

print(accuracy)