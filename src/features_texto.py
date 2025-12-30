import math
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.util import bigrams

nltk.download("stopwords")

STOPWORDS = set(stopwords.words("portuguese"))


def gerar_features_texto(texto: str) -> dict:
    palavras = texto.lower().split()
    total_palavras = len(palavras)

    if total_palavras == 0:
        return {
            "tamanho_texto": 0,
            "quantidade_palavras": 0,
            "tamanho_medio_palavra": 0,
            "diversidade_lexical": 0,
            "taxa_stopwords": 0,
            "entropia_texto": 0,
            "repeticao_bigramas": 0,
        }

    tamanho_texto = len(texto)
    palavras_unicas = set(palavras)

    tamanho_medio_palavra = sum(len(p) for p in palavras) / total_palavras
    diversidade_lexical = len(palavras_unicas) / total_palavras
    taxa_stopwords = sum(1 for p in palavras if p in STOPWORDS) / total_palavras

    frequencias = Counter(palavras)
    entropia = -sum(
        (freq / total_palavras) * math.log2(freq / total_palavras)
        for freq in frequencias.values()
    )

    lista_bigramas = list(bigrams(palavras))
    repeticao_bigramas = 0
    if lista_bigramas:
        repeticao_bigramas = 1 - (len(set(lista_bigramas)) / len(lista_bigramas))

    return {
        "tamanho_texto": tamanho_texto,
        "quantidade_palavras": total_palavras,
        "tamanho_medio_palavra": tamanho_medio_palavra,
        "diversidade_lexical": diversidade_lexical,
        "taxa_stopwords": taxa_stopwords,
        "entropia_texto": entropia,
        "repeticao_bigramas": repeticao_bigramas,
    }
