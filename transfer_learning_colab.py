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
test_size = total_size - train_size - val_size  # Garantir que o resto seja o conjunto de teste

# A divisão de dados do TensorFlow é um pouco complicada,
# uma abordagem mais segura é usar a divisão de subconjuntos
# de treinamento, validação e teste do próprio `tfds.load()`:
train_ds, val_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    split=[
        f"train[:{train_size}]",  # 80% para treino
        f"train[{train_size}:{train_size + val_size}]",  # 10% para validação
        f"train[{train_size + val_size}:]",  # 10% para teste
    ],
    as_supervised=True,
)

# ===============================
# 2. Pré-processamento
# ===============================
def format_example(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0  # normalização
    return image, label

# Aplicar pré-processamento e batch
train_ds = train_ds.map(format_example).shuffle(1000, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(format_example).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(format_example).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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
print(f"Acurácia no conjunto de teste: {acc:.4f}")

# ===============================
# 6. Visualização de métricas
# ===============================
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(EPOCHS)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Acurácia de Treino")
plt.plot(epochs_range, val_acc, label="Acurácia de Validação")
plt.legend(loc="lower right")
plt.title("Acurácia de Treino e Validação")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Loss de Treino")
plt.plot(epochs_range, val_loss, label="Loss de Validação")
plt.legend(loc="upper right")
plt.title("Loss de Treino e Validação")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.show()

# ===============================
# 7. Salvar modelo
# ===============================
os.makedirs("models", exist_ok=True)
model.save("models/transfer_learning_model.h5")
print("Modelo salvo em models/transfer_learning_model.h5")
