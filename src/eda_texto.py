import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt
from collections import Counter
import nltk

# tokenização robusta: tenta garantir recursos do NLTK e faz fallback simples
def _safe_word_tokenize(text):
    text = "" if text is None else str(text)
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
            return nltk.word_tokenize(text)
        except Exception:
            try:
                nltk.download("punkt_tab", quiet=True)
                return nltk.word_tokenize(text)
            except Exception:
                return text.split()

class AnaliseExploratoriaTexto:
    """
    Classe responsável por realizar análise exploratória
    em datasets de texto (Natty vs Fake Natty).
    """

    def __init__(self, caminho_dataset: str):
        """
        Inicializa a análise carregando o dataset.

        Espera colunas:
        - texto
        - label (1 = humano, 0 = IA)
        """
        try:
            self.df = pd.read_csv(caminho_dataset)
        except EmptyDataError:
            raise ValueError(f"Arquivo '{caminho_dataset}' está vazio. Coloque um CSV com colunas 'texto' e 'label'.")

        if self.df is None or self.df.empty:
            raise ValueError(f"Arquivo '{caminho_dataset}' está vazio ou não contém dados. Coloque um CSV com colunas 'texto' e 'label'.")

        expected = {"texto", "label"}
        if not expected.issubset(set(self.df.columns)):
            raise ValueError(f"Arquivo '{caminho_dataset}' deve conter as colunas: {expected}")

        self._preparar_dados()

    def _preparar_dados(self):
        """Cria colunas auxiliares para análise."""
        self.df["texto"] = self.df["texto"].astype(str)
        self.df["quantidade_palavras"] = self.df["texto"].apply(
            lambda x: len(_safe_word_tokenize(x))
        )
        self.df["tamanho_texto"] = self.df["texto"].apply(len)
        self.df["diversidade_lexical"] = self.df["texto"].apply(self._diversidade_lexical)

    @staticmethod
    def _diversidade_lexical(texto: str) -> float:
        palavras = _safe_word_tokenize(texto)
        if not palavras:
            return 0.0
        return len(set(palavras)) / len(palavras)

    # =========================
    # 📊 VISUALIZAÇÕES
    # =========================

    def plot_distribuicao_labels(self):
        """Distribuição Natty vs Fake Natty."""
        self.df["label"].value_counts().plot(
            kind="bar",
            title="Distribuição de Classes (0 = IA | 1 = Humano)"
        )
        plt.xlabel("Classe")
        plt.ylabel("Quantidade")
        plt.show()

    def plot_tamanho_texto(self):
        """Distribuição do tamanho dos textos."""
        self.df.boxplot(
            column="tamanho_texto",
            by="label",
            grid=False
        )
        plt.title("Tamanho do Texto por Classe")
        plt.suptitle("")
        plt.xlabel("Classe")
        plt.ylabel("Quantidade de Caracteres")
        plt.show()

    def plot_quantidade_palavras(self):
        """Quantidade de palavras por classe."""
        self.df.boxplot(
            column="quantidade_palavras",
            by="label",
            grid=False
        )
        plt.title("Quantidade de Palavras por Classe")
        plt.suptitle("")
        plt.xlabel("Classe")
        plt.ylabel("Palavras")
        plt.show()

    def plot_diversidade_lexical(self):
        """Diversidade lexical por classe."""
        self.df.boxplot(
            column="diversidade_lexical",
            by="label",
            grid=False
        )
        plt.title("Diversidade Lexical por Classe")
        plt.suptitle("")
        plt.xlabel("Classe")
        plt.ylabel("Diversidade")
        plt.show()

    # =========================
    # 🧠 ANÁLISES TEXTUAIS
    # =========================

    def palavras_mais_frequentes(self, label: int, top_n: int = 10):
        """
        Retorna as palavras mais frequentes para uma classe.
        label: 0 = IA | 1 = Humano
        """
        textos = self.df[self.df["label"] == label]["texto"]
        palavras = []

        for texto in textos:
            palavras.extend(_safe_word_tokenize(texto.lower()))

        contador = Counter(palavras)
        return contador.most_common(top_n)

    def resumo_estatistico(self):
        """Resumo estatístico por classe."""
        return self.df.groupby("label")[
            ["tamanho_texto", "quantidade_palavras", "diversidade_lexical"]
        ].describe()

    # =========================
    # 🔍 INSIGHTS
    # =========================

    def comparar_medias(self):
        """Compara médias entre textos humanos e IA."""
        return self.df.groupby("label")[
            ["tamanho_texto", "quantidade_palavras", "diversidade_lexical"]
        ].mean()

    def amostras_texto(self, label: int, n: int = 3):
        """Exibe exemplos de textos por classe."""
        return self.df[self.df["label"] == label]["texto"].sample(n, random_state=42)