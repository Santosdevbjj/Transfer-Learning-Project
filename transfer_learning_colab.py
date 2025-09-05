# transfer_learning_colab.py
# Projeto de Transfer Learning em Python
# Autor: Sérgio Santos
# Descrição: Exemplo de aplicação de Transfer Learning com MobileNetV2
# Dataset: Cats vs Dogs (pode ser substituído por dataset próprio)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds

# ===============================
# 1. Carregar dataset
# ===============================
print("Carregando dataset Cats vs Dogs...")
dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)

# Divisão entre treino e teste (se não houver split de teste, criamos manualmente)
train_ds = dataset['train']
test_ds = dataset['train'].take(2000)  # usa parte do dataset como teste

# ===============================
# 2. Pré-processamento
# ===============================
IMG_SIZE = (160, 160)

def format_example(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, label

train_ds = train_ds.map(format_example).batch(32).shuffle(1000)
test_ds = test_ds.map(format_example).batch(32)

# ===============================
# 3. Carregar modelo pré-treinado
# ===============================
print("Carregando MobileNetV2 pré-treinada no ImageNet...")
base_model = keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # congelar camadas base

# Construção do modelo
global_avg = layers.GlobalAveragePooling2D()(base_model.output)
output_layer = layers.Dense(1, activation='sigmoid')(global_avg)
model = models.Model(inputs=base_model.input, outputs=output_layer)

# ===============================
# 4. Compilar e Treinar
# ===============================
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Treinando modelo...")
history = model.fit(train_ds, validation_data=test_ds, epochs=3)

# ===============================
# 5. Avaliar desempenho
# ===============================
loss, acc = model.evaluate(test_ds)
print(f"Acurácia no conjunto de teste: {acc:.2f}")

# ===============================
# 6. Salvar modelo
# ===============================
model.save('transfer_learning_model.h5')
print("Modelo salvo como transfer_learning_model.h5")
