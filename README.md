## Treinamento de Redes Neurais com Transfer Learning.

![Machine001](https://github.com/user-attachments/assets/6e3d41d5-0b53-449c-b4e2-0d32b9036a07)


**Bootcamp BairesDev - Machine Learning Training**

--- 

**DESCRIÇÃO:**

O projeto consiste em aplicar o método de Transfer Learning em uma rede de Deep Learning na linguagem Python no ambiente COLAB. 


Neste projeto, você pode usar sua própria base de dados (exemplo: fotos suas, dos seus pais, dos seus amigos, dos seus animais domésticos, etc), o exemplo de gatos e cachorros, pode ser substituído por duas outras classes do seu interesse.


---


---

# 🧠 Transfer Learning em Python (TensorFlow + Keras)

Projeto acadêmico de **Treinamento de Redes Neurais com Transfer Learning**, desenvolvido em Python para rodar no **Google Colab** ou localmente.

## 🚀 Como funciona?
- Utilizamos **MobileNetV2** como backbone pré-treinado em ImageNet.
- O modelo é adaptado para classificar **duas classes** (ex.: gatos vs cachorros, ou seu próprio dataset).
- Inclui **data augmentation**, **fine-tuning opcional**, e exportação do modelo.

## 📂 Estrutura do Projeto
- `transfer_learning.py` → Script principal em Python.
- `Transfer_Learning_Colab.ipynb` → Notebook pronto para abrir no Colab.
- `data/` → Dataset organizado em `train/` e `val/`.
- `models/` → Modelos treinados.
- `results/` → Resultados (gráficos, logs, matriz de confusão).
- `docs/` → Relatório/documentação técnica.

## ▶️ Executando no Colab
Clique no botão abaixo para abrir no Google Colab:

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Santosdevbjj/Transfer-Learning-Project/blob/main/Transfer_Learning_Colab.ipynb)

## ⚙️ Requisitos
Instale as dependências:
```bash
pip install -r requirements.txt

