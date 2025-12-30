import re
import math
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class FeatureTexto:
    """
    Extrai features linguísticas para classificação
    Natty (humano) vs Fake Natty (IA)
    """

    def __init__(self, texto_coluna: str = "texto"):
        self.texto_coluna = texto_coluna

    # =========================
    # Utilitários
    # =========================
    def _tokenizar(self, texto: str):
        return re.findall(r"\b\w+\b", texto.lower())

    def _bigramas(self, tokens):
        return list(zip(tokens, tokens[1:]))

    # =========================
    # Features básicas
    # =========================
    def tamanho_texto(self, texto):
        return len(texto)

    def quantidade_palavras(self, tokens):
        return len(tokens)

    def tamanho_medio_palavra(self, tokens):
        if not tokens:
            return 0
        return sum(len(t) for t in tokens) / len(tokens)

    def variancia_tamanho_palavra(self, tokens):
        if not tokens:
            return 0
        tamanhos = [len(t) for t in tokens]
        media = sum(tamanhos) / len(tamanhos)
        return sum((t - media) ** 2 for t in tamanhos) / len(tamanhos)

    # =========================
    # Features linguísticas
    # =========================
    def diversidade_lexical(self, tokens):
        if not tokens:
            return 0
        return len(set(tokens)) / len(tokens)

    def taxa_stopwords(self, tokens):
        if not tokens:
            return 0
        stopwords = [t for t in tokens if t in ENGLISH_STOP_WORDS]
        return len(stopwords) / len(tokens)

    def entropia_texto(self, tokens):
        if not tokens:
            return 0
        contagem = Counter(tokens)
        total = len(tokens)
        entropia = 0
        for freq in contagem.values():
            p = freq / total
            entropia -= p * math.log2(p)
        return entropia

    def repeticao_bigramas(self, tokens):
        bigramas = self._bigramas(tokens)
        if not bigramas:
            return 0
        contagem = Counter(bigramas)
        repetidos = sum(1 for c in contagem.values() if c > 1)
        return repetidos / len(bigramas)

    def taxa_pontuacao(self, texto):
        if not texto:
            return 0
        pontuacao = re.findall(r"[.,!?;:]", texto)
        return len(pontuacao) / len(texto)

    # =========================
    # Pipeline principal
    # =========================
    def extrair(self, df: pd.DataFrame) -> pd.DataFrame:
        features = []

        for texto in df[self.texto_coluna]:
            tokens = self._tokenizar(texto)

            features.append({
                "tamanho_texto": self.tamanho_texto(texto),
                "quantidade_palavras": self.quantidade_palavras(tokens),
                "tamanho_medio_palavra": self.tamanho_medio_palavra(tokens),
                "variancia_tamanho_palavra": self.variancia_tamanho_palavra(tokens),
                "diversidade_lexical": self.diversidade_lexical(tokens),
                "taxa_stopwords": self.taxa_stopwords(tokens),
                "entropia_texto": self.entropia_texto(tokens),
                "repeticao_bigramas": self.repeticao_bigramas(tokens),
                "taxa_pontuacao": self.taxa_pontuacao(texto),
            })

        return pd.DataFrame(features)
