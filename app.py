import marimo

__generated_with = "0.13.11"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from transformers import pipeline

    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    return pd, pipeline, plt


@app.cell
def _(pipeline):
    classifier = pipeline(
        "text-classification",
        model="Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis",
    )
    return (classifier,)


@app.cell
def _(classifier):
    def predict_sentiment(text):
        results = classifier(text, return_all_scores=True)[0]
        return {r["label"]: round(r["score"], 4) for r in results}
    return (predict_sentiment,)


@app.cell
def _(pd):
    barak = pd.read_csv("barak.csv")
    barak.head()
    return (barak,)


@app.cell
def _(barak, pd, predict_sentiment):
    texts = barak["full_text"]

    results = []

    # Loop prediksi
    for text in texts:
        sentiment_scores = predict_sentiment(text)
        row = {"Text": text}
        row.update(sentiment_scores)
        # Ambil label dominan
        row["PredictedLabel"] = max(sentiment_scores, key=sentiment_scores.get)
        results.append(row)

    df = pd.DataFrame(results)

    # Optional: tampilkan 3 digit di belakang koma
    pd.options.display.float_format = "{:.3f}".format
    return (df,)


@app.cell
def _(df):
    df.sort_values(by="Neutral", ascending=False).tail()
    return


@app.cell
def _(df, plt):
    # Hitung frekuensi masing-masing label
    label_counts = df["PredictedLabel"].value_counts()

    # Visualisasi pie chart
    # colors = ["#FF6B6B", "#FFD93D", "#6BCB77"]  # merah, kuning, hijau
    plt.figure(figsize=(6, 6))
    plt.pie(
        label_counts,
        labels=label_counts.index,
        autopct="%1.1f%%",
        colors=["#FF6B6B", "#FFD93D", "#6BCB77"],
        startangle=140,
    )
    plt.title("Distribusi Sentimen Dominan")
    plt.axis("equal")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
