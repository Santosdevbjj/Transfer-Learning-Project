## Treinamento de Redes Neurais com Transfer Learning.

![Machine001](https://github.com/user-attachments/assets/6e3d41d5-0b53-449c-b4e2-0d32b9036a07)


**Bootcamp BairesDev - Machine Learning Training** 
 
--- 
📌 **Descrição:**


O projeto consiste em aplicar o método de Transfer Learning em uma rede de Deep Learning na linguagem Python no ambiente COLAB. 

Neste projeto, você pode usar sua própria base de dados (exemplo: fotos suas, dos seus pais, dos seus amigos, dos seus animais domésticos, etc), o exemplo de gatos e cachorros, pode ser substituído por duas outras classes do seu interesse.

---

Este projeto demonstra o uso de Transfer Learning em redes neurais utilizando Python e TensorFlow.
O objetivo é treinar uma rede de deep learning em um dataset personalizado (ou no dataset público de gatos vs cachorros) com Colab ou localmente.

A ideia central é aproveitar modelos pré-treinados como MobileNetV2 e adaptá-los a novas classes, reduzindo custo de treino e melhorando a performance em datasets pequenos.


---

# 🧠 Projeto de Transfer Learning em Python (TensorFlow/Keras)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Santosdevbjj/Transfer-Learning-Project/blob/main/transfer_learning_colab.py)

Este repositório implementa **Transfer Learning** com a rede **MobileNetV2**, pré-treinada no **ImageNet**, adaptada para classificação binária.  
O exemplo utiliza o dataset **Cats vs Dogs** do `tensorflow_datasets`, mas você pode facilmente substituir por suas próprias imagens.

---

# Transfer Learning Project

[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

---


---


## 📂 Estrutura do Repositório


Transfer-Learning-Project/ │── 📄 README.md                  # Documentação do projeto │── 📄 LICENSE                    # Licença do projeto (MIT) │── 📄 requirements.txt           # Dependências do projeto │── 📄 .gitignore                 # Arquivos ignorados pelo Git │── 📄 transfer_learning_colab.py # Script principal com Transfer Learning


---

---

## 🚀 Como Executar

### 🔹 Google Colab (Recomendado)
1. Clique no botão **"Open In Colab"** acima.  
2. Execute as células para treinar e avaliar o modelo.  

### 🔹 Localmente (Python 3.8+)
1. Clone este repositório:
   ```bash
   git clone https://github.com/Santosdevbjj/Transfer-Learning-Project.git
   cd Transfer-Learning-Project


---


2. **Crie e ative um ambiente virtual:**

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows



   ---


3. **Instale as dependências:**

pip install -r requirements.txt



---

4. **Execute o script:**

python transfer_learning_colab.py

---


📊 **Resultados**

Modelo baseado em MobileNetV2.

Treinamento padrão com 3 épocas (ajustável).

**Exemplo de saída:**

Acurácia no conjunto de teste: 0.87
Modelo salvo como transfer_learning_model.h5



---

🔧 **Dependências**

As principais dependências estão listadas em requirements.txt:

tensorflow
tensorflow-datasets
matplotlib
numpy

---


📌 **Como Usar com Seu Próprio Dataset**

Se você quiser treinar com suas próprias imagens, siga estes passos:

**1. Organize seus dados no formato:**

<img width="853" height="503" alt="Screenshot_20250906-030921" src="https://github.com/user-attachments/assets/565ccb1a-96d4-4cae-a30a-95009d331a4d" />



---


2. No script transfer_learning_colab.py, substitua a parte do carregamento do dataset por:


train_ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(160, 160),
    batch_size=32
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(160, 160),
    batch_size=32
)


---

3. Rode o treinamento novamente e o modelo será adaptado ao seu dataset.


---

🛑 **.gitignore Melhorado**

Certifique-se de que seu .gitignore contém:


# Ambientes virtuais
venv/
__pycache__/

# Modelos treinados
*.h5
*.keras

# Arquivos temporários
*.log
*.tmp

---

**requirements-dev.txt**


🔎 **Explicação dos pacotes:**

pytest / pytest-cov → para rodar testes unitários com cobertura.

flake8 → análise de estilo e linting.

black → formatação automática de código.

isort → organização automática de imports.

mypy → verificação estática de tipos (bom para manter qualidade).

types-tensorflow → stubs de tipagem para TensorFlow.

jupyter e notebook → opcionais, mas úteis se você quiser rodar .ipynb localmente.

---

🚀 **Como usar no repositório:**

Você pode instalar as dependências de desenvolvimento separadamente com:

pip install -r requirements-dev.txt

---


E rodar os checks automáticos antes de dar commit, por exemplo:

black .
flake8 .
mypy .
pytest

---

📌 **3. Instalar o pre-commit localmente**

Depois de instalar as dependências (pip install -r requirements-dev.txt), rode:

pre-commit install

---

📌 **4. Testar manualmente os hooks**

Você pode rodar os hooks manualmente em todos os arquivos já existentes:

pre-commit run --all-files


---


📜 **Licença**

Este projeto está sob a licença MIT (veja o arquivo LICENSE).

---

✅ **Agora seu repositório tem:**  
1. **Badge do Colab** para abrir direto.  
2. **`.gitignore` aprimorado** (ignora modelos, cache e venv).  
3. **Seção para usar dataset próprio** bem explicada.  


---

