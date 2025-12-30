from pathlib import Path
import matplotlib
matplotlib.use('Agg')

# localizar dataset
root = Path().resolve()
for _ in range(8):
    candidate = root / "data" / "processed" / "dataset.csv"
    if candidate.exists():
        dataset_path = candidate
        break
    root = root.parent
else:
    raise FileNotFoundError("Arquivo data/processed/dataset.csv não encontrado")

# adicionar raiz ao sys.path e importar
import sys
sys.path.append(str(root))
from src.eda_texto import AnaliseExploratoriaTexto

print('Dataset encontrado em:', dataset_path)
eda = AnaliseExploratoriaTexto(str(dataset_path))
print('Dataset carregado, shape:', eda.df.shape)
print('Top 3 palavras humanas:', eda.palavras_mais_frequentes(label=1, top_n=3))
print('Resumo estatístico (head):')
print(eda.resumo_estatistico().head())
