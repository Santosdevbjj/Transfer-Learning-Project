# transfer_learning_colab.py
# Projeto de Transfer Learning em Python
# Autor: Sérgio Santos
# Descrição: Exemplo de aplicação de Transfer Learning com MobileNetV2
# Dataset: Cats vs Dogs (pode ser substituído por dataset próprio)

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

# ===============================
# 0. Configurações iniciais
# ===============================
SEED = 42
tf.random.set_seed(SEED)

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10  # mais adequado que 3 para avaliação de desempenho

# ===============================
# 1. Carregar dataset
# ===============================
print("Carregando dataset Cats vs Dogs...")
dataset, info = tfds.load("cats_vs_dogs", with_info=True, as_supervised=True)

total_size = info.splits["train"].num_examples
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)

# Shuffle global para evitar viés
full_dataset = dataset["train"].shuffle(1000, seed=SEED)

# Split manual: 80% treino, 10% validação, 10% teste
train_ds = full_dataset.take(train_size)
val_ds = full_dataset.skip(train_size).take(val_size)
test_ds = full_dataset.skip(train_size + val_size)

# ===============================
# 2. Pré-processamento
# ===============================
def format_example(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0  # normalização
    return image, label

train_ds = train_ds.map(format_example).batch(BATCH_SIZE).shuffle(1000, seed=SEED)
val_ds = val_ds.map(format_example).batch(BATCH_SIZE)
test_ds = test_ds.map(format_example).batch(BATCH_SIZE)

# ===============================
# 3. Carregar modelo pré-treinado
# ===============================
print("Carregando MobileNetV2 pré-treinada no ImageNet...")
base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base_model.trainable = False  # congelar camadas base

# Construção do modelo
global_avg = layers.GlobalAveragePooling2D()(base_model.output)
dropout = layers.Dropout(0.2)(global_avg)  # regularização para evitar overfitting
output_layer = layers.Dense(1, activation="sigmoid")(dropout)
model = models.Model(inputs=base_model.input, outputs=output_layer)

# ===============================
# 4. Compilar e Treinar
# ===============================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print("Treinando modelo...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# ===============================
# 5. Avaliar desempenho
# ===============================
loss, acc = model.evaluate(test_ds)
print(f"Acurácia no conjunto de teste: {acc:.2f}")

# ===============================
# 6. Visualização de métricas
# ===============================
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Treino")
plt.plot(epochs_range, val_acc, label="Validação")
plt.legend(loc="lower right")
plt.title("Acurácia")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Treino")
plt.plot(epochs_range, val_loss, label="Validação")
plt.legend(loc="upper right")
plt.title("Loss")
plt.show()

# ===============================
# 7. Salvar modelo
# ===============================
os.makedirs("models", exist_ok=True)
model.save("models/transfer_learning_model.h5")
print("Modelo salvo em models/transfer_learning_model.h5")
