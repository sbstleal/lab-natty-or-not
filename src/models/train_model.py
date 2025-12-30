import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.model_utils import criar_pipeline
from src.features_texto import gerar_features_texto


ARTIFACTS_DIR = "src/artifacts"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "modelo_natty.pkl")
DATASET_PATH = "data/processed/dataset.csv"


def main():
    print("📥 Carregando dataset...")
    df = pd.read_csv(DATASET_PATH)

    print("🧪 Preparando dados...")
    if "tamanho_texto" not in df.columns:
        print("⚙️ Gerando features textuais...")
        features = df["texto"].apply(gerar_features_texto).apply(pd.Series)
        df = pd.concat([df, features], axis=1)

    X = df.drop(columns=["texto", "label"])
    X = X.select_dtypes(include=["number"])
    y = df["label"]

    if X.empty:
        raise ValueError("❌ Nenhuma feature numérica encontrada.")

    print("✂️ Separando treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🧠 Treinando modelo...")
    pipeline = criar_pipeline()
    pipeline.fit(X_train, y_train)

    print("💾 Salvando modelo...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print(f"✅ Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    main()
