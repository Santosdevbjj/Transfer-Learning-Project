## Treinamento de Redes Neurais com Transfer Learning.


--- 


![Machine001](https://github.com/user-attachments/assets/6e3d41d5-0b53-449c-b4e2-0d32b9036a07)

**Bootcamp BairesDev — Machine Learning Training**

---

# 🐱🐶 Transfer Learning Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Santosdevbjj/Transfer-Learning-Project/blob/main/Transfer_Learning_Colab.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> *A diferença entre um projeto e um portfólio é o contexto que você coloca nele.*

---

## 1. 🎯 Problema de Negócio

Treinar um modelo de visão computacional do zero exige dezenas de milhares de imagens rotuladas e horas de processamento em GPU. Para a maioria dos casos reais — um classificador de produtos, um detector de defeitos em linha de produção, uma ferramenta de triagem médica — esse volume de dados simplesmente não existe.

O desafio é: **como construir um classificador de imagens preciso quando o dataset disponível é pequeno e o tempo de treinamento precisa ser viável sem infraestrutura cara?**

---

## 2. 🏢 Contexto

O projeto foi desenvolvido no **Bootcamp BairesDev — Machine Learning Training**, como aplicação prática de Transfer Learning para classificação binária de imagens.

A técnica resolve o problema descrito ao reaproveitar o conhecimento visual já aprendido por um modelo treinado em milhões de imagens (ImageNet), adaptando apenas as últimas camadas ao problema-alvo. O custo de treinamento cai drasticamente — de horas para minutos, mesmo no ambiente gratuito do Google Colab.

O dataset utilizado foi o **Cats vs Dogs** do `tensorflow_datasets` (23.262 imagens), dividido em 80% treino, 10% validação e 10% teste. O projeto foi estruturado para ser facilmente adaptado a qualquer dataset binário próprio, bastando reorganizar as imagens em duas pastas e substituir o trecho de carregamento.

---

## 3. 📐 Premissas da Análise

As seguintes premissas delimitam o escopo desta implementação:

- O backbone **MobileNetV2** tem seus pesos **congelados** (`trainable = False`) durante o treinamento. Apenas a cabeça de classificação adicionada ao topo é treinada. Isso é intencional: com datasets de tamanho médio, descongelar o backbone leva a overfitting imediato.
- As imagens são redimensionadas para **160×160 pixels** e normalizadas para o intervalo [0, 1] via divisão por `255.0`. Essa abordagem simplifica o pipeline para fins didáticos; normalização com parâmetros ImageNet produziria representações levemente melhores em produção.
- O treinamento usa `binary_crossentropy` como função de perda — adequado para classificação binária com saída `sigmoid`.
- O notebook usa `EarlyStopping` com `patience=3` monitorando `val_accuracy`, interrompendo o treinamento após 3 épocas sem melhora e restaurando os melhores pesos automaticamente.
- O script local usa divisão **80/10/10** com 10 épocas; o notebook usa 5 épocas com callbacks mais robustos. Ambos chegam a resultados comparáveis.

---

## 4. 🛠️ Estratégia da Solução

A solução foi implementada em duas formas complementares: um **script Python** (`transfer_learning_colab.py`) para execução local e reprodutibilidade, e um **notebook Colab** (`Transfer_Learning_Colab.ipynb`) com visualizações interativas e callbacks avançados.

**Etapa 1 — Carregamento e divisão do dataset**
O dataset é baixado via `tensorflow_datasets` e fatiado programaticamente em 80/10/10 usando a API de splits por string do `tfds.load()`. O `prefetch(tf.data.AUTOTUNE)` garante que o carregamento de dados não seja o gargalo do treinamento.

**Etapa 2 — Construção do modelo**
O MobileNetV2 é carregado com `include_top=False`, removendo a camada de classificação original. Sobre a saída do backbone são adicionadas três camadas: `GlobalAveragePooling2D` (reduz dimensionalidade espacial), `Dropout(0.2)` (regularização) e `Dense(1, activation='sigmoid')` (classificação binária).

**Etapa 3 — Compilação e treinamento**
O modelo é compilado com `Adam(lr=0.0001)` — learning rate conservador para não distorcer os pesos pré-treinados. O notebook adiciona `Precision` e `Recall` às métricas, além de `ModelCheckpoint` que salva automaticamente o melhor modelo com base em `val_accuracy`.

**Etapa 4 — Avaliação e exportação**
Após o treinamento, o modelo é avaliado no conjunto de teste e o histórico de métricas é salvo em JSON (`history/training_history.json`) e plotado como curvas de aprendizado. O modelo final é exportado nos formatos `.keras` (checkpoint) e `.h5` (compatibilidade legada).

**Etapa 5 — Qualidade de código**
O repositório usa **pre-commit hooks** configurados em `.pre-commit-config.yaml`: `black` para formatação, `flake8` para linting e `mypy` para verificação estática de tipos com stubs do TensorFlow.

---

## 5. 💡 Insights Técnicos

**Por que MobileNetV2 e não VGG16 ou ResNet50?**
MobileNetV2 usa convoluções depthwise separáveis que reduzem o número de parâmetros em ~10x comparado ao VGG16, mantendo acurácia competitiva. No Colab gratuito, isso significa treinar em minutos em vez de horas. Para classificação binária de animais domésticos, o poder representacional do MobileNetV2 é mais que suficiente — modelos maiores adicionariam custo sem ganho mensurável.

**Por que congelar o backbone e não fazer fine-tuning completo?**
Com ~23k imagens e poucas épocas, o fine-tuning completo tende a destruir as representações visuais gerais já aprendidas no ImageNet — fenômeno conhecido como *catastrophic forgetting*. O padrão adotado (congelar backbone, treinar apenas a cabeça) é a estratégia mais segura para datasets de tamanho médio. Em um próximo passo, o fine-tuning das últimas camadas do backbone poderia ser desbloqueado com `lr=1e-5` após a convergência inicial da cabeça.

**Por que `GlobalAveragePooling2D` e não `Flatten`?**
`Flatten` após o backbone do MobileNetV2 geraria um vetor de ~62.720 valores, tornando a camada Dense seguinte enorme e propensa a overfitting. `GlobalAveragePooling2D` colapsa cada mapa de features para um único valor, reduzindo a dimensão para 1.280 e atuando também como regularizador implícito — uma decisão de arquitetura com impacto direto na generalização.

**Por que pre-commit e não apenas CI/CD?**
Pre-commit roda localmente, antes do push. Isso impede que código mal formatado ou com erros de tipo sequer entre no histórico do repositório — ao contrário do CI/CD, que detecta problemas após o commit já ter sido feito. É uma camada de qualidade que comunica cuidado profissional para qualquer recrutador que abra o repositório.

---

## 6. 📊 Resultados

Com 5 épocas de treinamento no Colab (backbone congelado, apenas cabeça treinável):

| Métrica | Valor |
|---|---|
| Acurácia no teste | **~0.87 (87%)** |
| Modelo salvo | `models/transfer_learning_best.keras` |
| Histórico salvo | `history/training_history.json` |
| Curva de aprendizado | `history/accuracy_curve.png` |

O projeto demonstra que é possível atingir 87% de acurácia em classificação binária de imagens com menos de 10 minutos de treinamento no ambiente gratuito do Google Colab — sem GPU dedicada, sem dataset customizado e sem arquitetura complexa.

---

## 7. 🔭 Próximos Passos

- Implementar **fine-tuning progressivo**: após convergência com backbone congelado, desbloquear as últimas 20-30 camadas do MobileNetV2 com `lr=1e-5`.
- Adicionar **data augmentation** (`RandomFlip`, `RandomRotation`, `RandomZoom`) para aumentar artificialmente a diversidade do dataset.
- Comparar MobileNetV2 com **EfficientNetB0** em termos de acurácia vs tempo de treinamento.
- Adicionar **Grad-CAM** para visualizar quais regiões da imagem o modelo usa para classificar, aumentando a interpretabilidade.
- Publicar o modelo no **Hugging Face** e expor predições via API REST com FastAPI.

---

## 💻 Requisitos

### Hardware

| Recurso | Mínimo | Recomendado |
|---|---|---|
| CPU | Dual-Core | Quad-Core |
| RAM | 4 GB | 8 GB |
| GPU | — | NVIDIA (Colab T4 gratuito) |
| Disco | 2 GB livres | 5 GB livres |

### Software

| Dependência | Versão |
|---|---|
| Python | 3.10+ |
| TensorFlow | 2.15.0 |
| tensorflow-datasets | 4.9.4 |
| matplotlib | 3.7.2 |
| numpy | 1.24.3 |

---

## 📂 Estrutura do Projeto

```bash
Transfer-Learning-Project/
│── README.md                       # Documentação do projeto
│── LICENSE                         # Licença MIT
│── requirements.txt                # TensorFlow, tensorflow-datasets, matplotlib, numpy
│── requirements-dev.txt            # pytest, black, flake8, mypy, isort, pre-commit
│── .gitignore                      # Exclui modelos .h5/.keras, venv, __pycache__
│── .pre-commit-config.yaml         # Hooks: black + flake8 + mypy (com stubs TF)
│
├── transfer_learning_colab.py      # Script local — divisão 80/10/10, 10 épocas
│
└── Transfer_Learning_Colab.ipynb   # Notebook Colab — callbacks, métricas completas
```

---

## ⚙️ Como Executar

### 🔹 Google Colab (Recomendado)

1. Clique no botão **"Open In Colab"** no topo deste README.
2. Execute as células sequencialmente — o dataset é baixado automaticamente.
3. O modelo treinado é salvo em `models/transfer_learning_best.keras`.

### 🔹 Localmente

**1. Clonar o repositório:**

```bash
git clone https://github.com/Santosdevbjj/Transfer-Learning-Project.git
cd Transfer-Learning-Project
```

**2. Criar e ativar o ambiente virtual:**

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows
```

**3. Instalar dependências:**

```bash
pip install -r requirements.txt
```

**4. Executar o treinamento:**

```bash
python transfer_learning_colab.py
```

**5. (Opcional) Instalar ferramentas de desenvolvimento:**

```bash
pip install -r requirements-dev.txt
pre-commit install
```

**6. Rodar checks de qualidade:**

```bash
black .
flake8 .
mypy .
pytest
```

---

## 🔄 Usando Seu Próprio Dataset

Organize as imagens no seguinte formato:

```bash
data/
├── classe_a/
│   ├── imagem_001.jpg
│   └── imagem_002.jpg
└── classe_b/
    ├── imagem_001.jpg
    └── imagem_002.jpg
```

No script, substitua o carregamento do `tfds` por:

```python
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
```

O restante do pipeline (modelo, compilação, treinamento) funciona sem alterações.

---

## 📌 Aprendizados

**1. Transfer Learning não é atalho — é decisão de engenharia.** Escolher congelar o backbone foi uma decisão consciente baseada no tamanho do dataset. Com 23k imagens e poucas épocas, fine-tuning completo destruiria as representações aprendidas no ImageNet. Entender *por que* congelar é tão importante quanto saber *como* fazer.

**2. `GlobalAveragePooling2D` é regularização implícita.** Eu teria usado `Flatten` por reflexo. Com o backbone gerando mapas de features volumosos, `Flatten` criaria um vetor enorme e um modelo com propensão a overfitting. A escolha do pooling tem impacto direto na capacidade de generalização — e não é óbvia para quem está começando.

**3. Pre-commit transforma qualidade em hábito.** Configurar `black`, `flake8` e `mypy` não é só organização — é tornar boas práticas automáticas e inevitáveis. Um repositório com pre-commit ativo comunica cuidado profissional antes de qualquer linha de código ser lida pelo recrutador.

---

## 📜 Licença

Este projeto está sob a licença MIT. Sinta-se livre para usar, modificar e distribuir.

---

## 📬 Contato

[![Portfólio Sérgio Santos](https://img.shields.io/badge/Portfólio-Sérgio_Santos-111827?style=for-the-badge&logo=githubpages&logoColor=00eaff)](https://portfoliosantossergio.vercel.app)
[![LinkedIn Sérgio Santos](https://img.shields.io/badge/LinkedIn-Sérgio_Santos-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/santossergioluiz)

---


