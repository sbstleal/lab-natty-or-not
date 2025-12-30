import pandas as pd
from pathlib import Path
import sys


def carregar_textos(caminho, label):
    with open(caminho, "r", encoding="utf-8") as f:
        textos = f.readlines()

    return pd.DataFrame({
        "texto": [t.strip() for t in textos if t.strip()],
        "label": label
    })


output_path = Path("data/processed/dataset.csv")
if output_path.exists():
    print(f"Arquivo '{output_path}' já existe. Não vou sobrescrever. Remova o arquivo ou use outro caminho.")
    sys.exit(0)


df_humano = carregar_textos("data/raw/textos_humanos.txt", 1)
df_ia = carregar_textos("data/raw/textos_ia.txt", 0)

df = pd.concat([df_humano, df_ia], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# garantir diretório de saída
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("dataset.csv criado com sucesso em:", output_path)
