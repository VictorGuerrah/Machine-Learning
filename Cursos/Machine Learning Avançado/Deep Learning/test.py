import tensorflow as tf
from tensorflow import keras


raw_dataset = keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = raw_dataset.load_data()

print(X_train)
