# Natural ou Fake Natty? 🤖🧠  
### Como Vencer na Era das IAs Generativas

> Projeto desenvolvido a partir do **fork oficial do Lab “Natty or Not” da DIO**, idealizado pelo professor **Venilton (falvojr)**.

---

## 🚀 Introdução

Inspirado na tendência **“Natty or Not”** do fisiculturismo, este projeto explora o universo das **IAs Generativas** sob uma perspectiva prática e crítica.

O desafio proposto é responder à seguinte pergunta:

👉 **É possível identificar, por meio de padrões linguísticos, se um texto foi escrito por um humano (*Natty*) ou gerado por uma Inteligência Artificial (*Fake Natty*)?**

Este repositório apresenta uma solução experimental baseada em **Processamento de Linguagem Natural (NLP)** e **Machine Learning**, com foco em **features estatísticas e interpretáveis**.

---

## 🎯 Objetivo do Projeto

- Explorar o uso de IA de forma crítica e consciente
- Analisar diferenças linguísticas entre textos humanos e textos gerados por IA
- Desenvolver um **classificador Natty vs Fake Natty**
- Aplicar conceitos de NLP, Engenharia de Features e Modelagem Supervisionada
- Fortalecer o portfólio acadêmico e profissional

---

## 📒 Descrição

O projeto consiste na construção de um **classificador supervisionado** capaz de estimar se um texto possui maior probabilidade de ter sido escrito por um humano ou por uma IA.

A abordagem adotada **não utiliza grandes modelos generativos para a classificação**, mas sim **métricas linguísticas e estatísticas**, como diversidade lexical, entropia e padrões de repetição, tornando o processo mais **explicável e reproduzível**.

---

## 🤖 Tecnologias Utilizadas

### 🧠 Inteligência Artificial & NLP
- NLTK
- Scikit-learn

### 🛠️ Ferramentas e Bibliotecas
- Python 3.10+
- Pandas
- NumPy
- Joblib
- Matplotlib
- Jupyter Notebook
- Git & GitHub

---

## 🧐 Processo de Criação

### 1. Fork do Repositório Oficial
- Fork do Lab **Natty or Not** disponibilizado pela DIO

### 2. Construção do Dataset
- Textos escritos por humanos
- Textos gerados por IA
- Rotulagem binária:
  - `0` → Fake Natty (IA)
  - `1` → Natty (Humano)

### 3. Análise Exploratória de Dados (EDA)
- Distribuição das classes
- Estatísticas descritivas dos textos
- Comparação entre padrões linguísticos

### 4. Engenharia de Features Textuais
Extração de características como:
- Tamanho do texto
- Quantidade de palavras
- Tamanho médio das palavras
- Diversidade lexical
- Taxa de stopwords
- Entropia do texto
- Repetição de bigramas

### 5. Modelagem Supervisionada
- Pipeline com:
  - Padronização dos dados (StandardScaler)
  - Regressão Logística
- Treinamento e salvamento do modelo
- Avaliação com métricas clássicas de classificação

---

## 🚀 Como Executar o Projeto

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/sbstleal/lab-natty-or-not.git
cd lab-natty-or-not

### 2️⃣ Criar e ativar o ambiente virtual

Windows (PowerShell):

    python -m venv .venv
    .venv\Scripts\Activate

Linux / Mac:

    python -m venv .venv
    source .venv/bin/activate

---

### 3️⃣ Instalar dependências

    pip install -r requirements.txt

---

### 4️⃣ Treinar o modelo

    python -m src.models.train_model

O modelo treinado será salvo automaticamente em:

    src/artifacts/modelo_natty.pkl

---

### 5️⃣ Avaliar o modelo

    python -m src.models.evaluate_model

A avaliação apresenta:
- Acurácia
- Precision
- Recall
- F1-score por classe

## 📊 Resultados

Com um dataset reduzido e de caráter experimental, o modelo obteve:

- **Acurácia aproximada:** 66%
- Boa identificação de textos gerados por IA
- Limitações esperadas na classe humana devido ao volume reduzido de dados

📌 Os resultados validam a **abordagem metodológica**, não representando um modelo final de produção.

---

## 💭 Reflexão

Criar algo verdadeiramente **“Natty”** na era das IAs Generativas é um desafio cada vez maior.

Embora modelos de IA consigam produzir textos extremamente realistas, ainda é possível identificar **padrões estatísticos sutis** que diferenciam textos humanos de textos artificiais — desde que se aceite que o resultado será sempre **probabilístico**, nunca absoluto.

Este projeto reforça a importância do uso **consciente, ético e transparente** da Inteligência Artificial.

## 🔗 Créditos e Referências

- Projeto original: **Lab Natty or Not – DIO**
- Professor: [Venilton (falvojr)](https://www.linkedin.com/in/falvojr)
- DIO: [Digital Innovation One](https://www.linkedin.com/school/dio-makethechange)

📌 Hashtag do desafio: **#LabDIONattyOrNot**