from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

model = LinearSVC(dual=True)

#Devo aumentar o tempo de campanha?
# 1 => aumentar
# 0 => parar
#Casos em que devo aumentar o tempo de campanha:

# meu dinheiro da para expandir a campanha ou é limitado?
# meu publico alvo ta numa crescente ou decrescente?
# minha frequencia está saudavel?

c1 = [1, 1, 1]
c2 = [1, 0, 1]
c3 = [0, 1, 1]

#Casos em que devo interromper o tempo de campanha:
c4 = [1, 0, 0]
c5 = [0, 0, 1]
c6 = [0, 1, 0]


train_x = [c1, c2, c3, c4, c5, c6] # dados de treino
train_y = [1, 1, 1, 0, 0, 0] # classes de treino - labels
model.fit(train_x, train_y)

test_x = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]
test_y = [0, 1, 1]

predictions = model.predict(test_x)
accuracy = accuracy_score(test_y, predictions)

print(accuracy)