import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from keras import layers, models, metrics, optimizers
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

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

num_features = 300
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(num_features,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


α = 0.01
σ = 1.0
class_weights_list = [1, 0.5]
max_epochs = 20

model.compile(optimizer=keras.optimizers.Adam(learning_rate=α), loss='binary_crossentropy', metrics=['accuracy'])

def train_model(X_train, y_train, α, σ, max_epochs, batch_size):
    percent_selected = 0.45
    num_samples = X_train.shape[0]
    num_selected = int(percent_selected * num_samples)
    selected_indices = np.random.choice(num_samples, num_selected, replace=False)

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        y_pred_train = []
        y_true_train = []
        epoch_losses = []
        weighted_epoch_losses = []
        shuffled_indices = np.random.permutation(num_samples)

        for batch_start in range(0, num_samples, batch_size):
            batch_indices = shuffled_indices[batch_start:batch_start + batch_size]
            selected_batch_indices = [i for i in batch_indices if i in selected_indices]
            noisy_batch_indices = [i for i in batch_indices if i not in selected_indices]
            selected_batch_X = X_train[selected_batch_indices]
            selected_batch_y = y_train[selected_batch_indices]
            selected_batch_y = tf.reshape(selected_batch_y, (-1, 1))
            noisy_batch_X = X_train[noisy_batch_indices]
            noisy_batch_y = y_train[noisy_batch_indices]
            noisy_batch_y = tf.reshape(noisy_batch_y, (-1, 1))

            with tf.GradientTape() as tape_selected:
                y_pred_selected = model(selected_batch_X)
                selected_batch_y = tf.cast(selected_batch_y, dtype=tf.float32)
                loss_selected = tf.keras.losses.binary_crossentropy(selected_batch_y, y_pred_selected)
                class_weights_tensor = tf.convert_to_tensor(class_weights_list, dtype=tf.float32)
                weighted_loss_selected = loss_selected * tf.gather(class_weights_tensor, tf.cast(selected_batch_y, dtype=tf.int32))
                average_weighted_loss_selected = tf.reduce_mean(weighted_loss_selected)

            gradients_selected = tape_selected.gradient(average_weighted_loss_selected, model.trainable_variables)
            noisy_gradients_selected = [g + tf.random.normal(g.shape, mean=0.0, stddev=σ)
                                        for g in gradients_selected]

            optimizer.apply_gradients(zip(noisy_gradients_selected, model.trainable_variables))

            y_pred_train.extend(y_pred_selected.numpy().flatten())
            y_true_train.extend(selected_batch_y.numpy())
            epoch_losses.append(tf.reduce_mean(loss_selected).numpy())
            weighted_epoch_losses.append(tf.reduce_mean(weighted_loss_selected).numpy())

            with tf.GradientTape() as tape_noisy:
                y_pred_noisy = model(noisy_batch_X)
                noisy_batch_y = tf.cast(noisy_batch_y, dtype=tf.float32)
                loss_noisy = tf.keras.losses.binary_crossentropy(noisy_batch_y, y_pred_noisy)
                class_weights_tensor = tf.convert_to_tensor(class_weights_list, dtype=tf.float32)
                weighted_loss_noisy = loss_noisy * tf.gather(class_weights_tensor, tf.cast(selected_batch_y, dtype=tf.int32))
                average_weighted_loss_noisy = tf.reduce_mean(weighted_loss_noisy)

            gradients_noisy = tape_noisy.gradient(average_weighted_loss_noisy, model.trainable_variables)

            # Update model parameters for noisy samples (without noise)
            optimizer.apply_gradients(zip(gradients_noisy, model.trainable_variables))
            y_pred_train.extend(y_pred_noisy.numpy().flatten())
            y_true_train.extend(noisy_batch_y.numpy())
            epoch_losses.append(tf.reduce_mean(loss_noisy).numpy())

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

batch_size = 64
train_model(X_train, y_train, α, σ, max_epochs, batch_size)

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

# Calculate metrics for the test set
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
