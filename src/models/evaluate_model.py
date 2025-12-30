import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from src.features_texto import gerar_features_texto


MODEL_PATH = "src/artifacts/modelo_natty.pkl"
DATASET_PATH = "data/processed/dataset.csv"


def main():
    print("📥 Carregando dataset...")
    df = pd.read_csv(DATASET_PATH)

    print("⚙️ Gerando features...")
    features = df["texto"].apply(gerar_features_texto).apply(pd.Series)
    X = features
    y = df["label"]

    print("📦 Carregando modelo...")
    model = joblib.load(MODEL_PATH)

    print("📊 Avaliando modelo...")
    y_pred = model.predict(X)

    print("Acurácia:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))


if __name__ == "__main__":
    main()
