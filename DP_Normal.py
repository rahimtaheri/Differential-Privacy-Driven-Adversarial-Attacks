import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from keras import layers, models, metrics, optimizers
import pickle

# Load the dataset
with open('./dataset/trainset.pkl', 'rb') as file:
    trainset = pickle.load(file)
with open('./dataset/testset.pkl', 'rb') as file:
    testset = pickle.load(file)

X_train = trainset[:, :-1]
y_train = trainset[:, -1]

X_test = testset[:, :-1]
y_test = testset[:, -1]

print("X_train_size: ", len(X_train))
print("X_test_size: ", len(X_test))

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(300,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'))
])

# Set hyperparameters
learning_rate = 0.01  
class_weights = {0: 1 , 1: 0.5}

def calculate_fpr(y_true, y_pred):
    threshold = 0.5
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    fp = tf.reduce_sum(tf.cast(tf.logical_and(y_true == 0, y_pred_binary == 1), tf.float32))
    tn = tf.reduce_sum(tf.cast(tf.logical_and(y_true == 0, y_pred_binary == 0), tf.float32))
    fpr = fp / (fp + tn)
    return fpr

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=[
                  metrics.BinaryAccuracy(),
                  metrics.Precision(),
                  metrics.Recall(),
                  metrics.Recall(),
                  metrics.FalsePositives(),
                  metrics.FalseNegatives(),
                  metrics.TruePositives(),
                  metrics.TrueNegatives(),
                  calculate_fpr])

model.fit(X_train, y_train, epochs=20, batch_size=64, class_weight=class_weights)

# Evaluate the model on the test set
test_results = model.evaluate(X_test, y_test)
test_loss = test_results[0]  
test_accuracy = test_results[1]  
y_pred = model.predict(X_test)
y_pred_binary = tf.keras.backend.round(y_pred).numpy()
y_pred_binary = y_pred_binary.flatten()
y_test_binary = y_test

tp = tf.keras.backend.sum(y_test_binary * y_pred_binary)
tn = tf.keras.backend.sum((1 - y_test_binary) * (1 - y_pred_binary))
fp = tf.keras.backend.sum((1 - y_test_binary) * y_pred_binary)
fn = tf.keras.backend.sum(y_test_binary * (1 - y_pred_binary))
precision = tp / (tp + fp + tf.keras.backend.epsilon())
recall = tp / (tp + fn + tf.keras.backend.epsilon())
fpr = fp / (fp + tn + tf.keras.backend.epsilon())

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Precision: {precision.numpy() * 100:.2f}%")
print(f"Recall: {recall.numpy() * 100:.2f}%")
print(f"FPR: {fpr.numpy() * 100:.2f}%")
print(f"False Positives (FP): {fp.numpy()}")
print(f"False Negatives (FN): {fn.numpy()}")
print(f"True Positives (TP): {tp.numpy()}")
print(f"True Negatives (TN): {tn.numpy()}")
