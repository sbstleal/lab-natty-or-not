import pandas as pd

def carregar_textos(caminho, label):
    with open(caminho, "r", encoding="utf-8") as f:
        textos = f.readlines()
    return pd.DataFrame({
        "texto": textos,
        "label": label
    })

df_humano = carregar_textos("data/raw/textos_humanos.txt", 1)
df_ia = carregar_textos("data/raw/textos_ia.txt", 0)

df = pd.concat([df_humano, df_ia]).sample(frac=1).reset_index(drop=True)
df.to_csv("data/processed/dataset.csv", index=False)
