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
    keras.layers.Dense(1, activation='sigmoid')
])

epsilon = 1.0
sensitivity = 1.0
learning_rate = 0.01
Nep = 10
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


def AdaptiveDPNoiseInjection(M, x, epsilon, sensitivity, percent_selected=0.45):
    num_samples = x.shape[0]
    num_selected = int(percent_selected * num_samples)
    selected_indices = np.random.choice(num_samples, num_selected, replace=False)
    selected_data = x[selected_indices]
    Us = np.random.rand(selected_data.shape[0], selected_data.shape[1])
    noise = -sensitivity / epsilon * np.sign(Us - 0.5) * np.log(1 - 2 * np.abs(Us - 0.5))
    selected_data_adv = selected_data + noise
    xadv = x.copy()
    xadv[selected_indices] = selected_data_adv
    return xadv

def check_differential_privacy(xadv,sensitivity, epsilon):
    privacy_loss = np.exp(epsilon) - 1
    p_violation = np.exp(-privacy_loss * sensitivity)
    random_value = np.random.rand()
    if random_value < p_violation:
        return False
    else:
        return True

poisoned_data = AdaptiveDPNoiseInjection(model, X_train, epsilon, sensitivity)

model.fit(poisoned_data, y_train, epochs=20, batch_size=64, class_weight=class_weights)

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

############################ Defence section ############################
def DifferentiallyPrivateNoise(ε, δ, N):
    laplace_noise = np.random.laplace(0, 2 / (ε * N), size=N)
    B = 1.0
    clipped_noise = np.clip(laplace_noise, -B / ε, B / ε)
    random_noise = np.random.laplace(0, B / ε, size=N)
    noisy_value = clipped_noise + random_noise
    return noisy_value

def train_model_Defence (X_train, y_train, α, ε, δ, max_epochs, batch_size):
    N= len(X_train)
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        y_pred_train = []
        y_true_train = []
        epoch_losses = []
        weighted_epoch_losses = []
        shuffled_indices = np.random.permutation(N)

        for batch_start in range(0, N, batch_size):
            batch_indices = shuffled_indices[batch_start:batch_start + batch_size]
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            batch_y = tf.reshape(batch_y, (-1, 1))
            with tf.GradientTape() as tape:
                y_pred = model(batch_X)
                batch_y = tf.cast(batch_y, dtype=tf.float32)
                loss = tf.keras.losses.binary_crossentropy(batch_y, y_pred)
                class_weights_tensor = tf.convert_to_tensor(class_weights_list, dtype=tf.float32)
                weighted_loss = loss * tf.gather(class_weights_tensor, tf.cast(batch_y, dtype=tf.int32))
                average_weighted_loss = tf.reduce_mean(weighted_loss)

            gradients = tape.gradient(average_weighted_loss, model.trainable_variables)
            Dpn= DifferentiallyPrivateNoise(ε, δ, N)
            noisy_gradients = [gi + dpi for gi, dpi in zip(gradients, Dpn)]
            optimizer.apply_gradients(zip(noisy_gradients, model.trainable_variables))
            y_pred_train.extend(y_pred.numpy().flatten())
            y_true_train.extend(batch_y.numpy())
            epoch_losses.append(tf.reduce_mean(loss).numpy())
            weighted_epoch_losses.append(tf.reduce_mean(weighted_loss).numpy())

        average_loss = np.mean(epoch_losses)
        weighted_epoch_losses = [l for l in weighted_epoch_losses if not np.isnan(l)]
        average_weighted_loss = np.mean(weighted_epoch_losses)
        y_pred_binary_train = [1 if pred >= 0.5 else 0 for pred in y_pred_train]

        accuracy = accuracy_score(y_true_train, y_pred_binary_train)
        precision = precision_score(y_true_train, y_pred_binary_train)
        recall = recall_score(y_true_train, y_pred_binary_train)
        conf_matrix = confusion_matrix(y_true_train, y_pred_binary_train, labels=[0, 1])
        tn, fp, fn, tp = conf_matrix.ravel()
        fpr = fp / (fp + tn + tf.keras.backend.epsilon())
        print(f"Accuracy: {accuracy:.4f} | Loss: {average_weighted_loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | FPR: {fpr:.4f} | FP: {fp} | TP: {tp} | FN: {fn} | TN: {tn}")


α = 0.01 
ε = 1 
δ = 1e-5
batch_size = 64
max_epochs = 20

optimizer= tf.keras.optimizers.Adam(learning_rate=α)
train_model_Defence(X_train, y_train, α, ε, δ, max_epochs, batch_size)

# Evaluate the model on the testing set
y_pred_test = []
test_losses = []

for batch_start in range(0, len(X_test), batch_size):
    batch_end = batch_start + batch_size
    batch_X = X_test[batch_start:batch_end]
    batch_y_pred = model.predict(batch_X)
    batch_test_losses = tf.keras.losses.binary_crossentropy(y_test[batch_start:batch_end], batch_y_pred)
    y_pred_test.append(batch_y_pred)
    test_losses.append(batch_test_losses)

y_pred_test = tf.concat(y_pred_test, axis=0)
test_losses = tf.concat(test_losses, axis=0)
average_test_loss = tf.reduce_mean(test_losses)
y_pred_test_binary = [1 if pred >= 0.5 else 0 for pred in y_pred_test.numpy()]
accuracy_test = accuracy_score(y_test, y_pred_test_binary)
precision_test = precision_score(y_test, y_pred_test_binary)
recall_test = recall_score(y_test, y_pred_test_binary)
conf_matrix_test = confusion_matrix(y_test, y_pred_test_binary, labels=[0, 1])
tn_test, fp_test, fn_test, tp_test = conf_matrix_test.ravel()
fpr_test = fp_test / (fp_test + tn_test + tf.keras.backend.epsilon())

print(f"Test Accuracy: {accuracy_test:.4f}")
print(f"Average Test Loss: {average_test_loss:.4f}")
print(f"Test Precision: {precision_test:.4f}")
print(f"Test Recall: {recall_test:.4f}")
print(f"Test FPR: {fpr_test:.4f}")
print(f"Test TP: {tp_test}")
print(f"Test FP: {fp_test}")
print(f"Test FN: {fn_test}")
print(f"Test TN: {tn_test}")