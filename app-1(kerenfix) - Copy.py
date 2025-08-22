from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route("/", methods=["GET", "POST"])
def index():
    # Baca data siswa
    df = pd.read_csv("data.csv")

    # Hitung inertia untuk elbow method
    X = df[["Tinggi", "Berat"]]
    inertias = []
    silhouette_scores = []
    K = range(2, 8)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append((k, round(silhouette_score(X, labels), 3)))

    # Buat grafik elbow
    plt.figure(figsize=(5, 3))
    plt.plot(K, inertias, marker="o")
    plt.title("Metode Elbow")
    plt.xlabel("Jumlah Cluster (k)")
    plt.ylabel("Inertia")
    elbow_plot = plot_to_base64()
    plt.close()

    # Ambil jumlah cluster dari dropdown (default = 3)
    n_clusters = int(request.form.get("n_clusters", 3))

    # Jalankan KMeans sesuai pilihan
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X)

    # Buat grafik clustering
    plt.figure(figsize=(5, 3))
    plt.scatter(df["Tinggi"], df["Berat"], c=df["Cluster"], cmap="viridis", s=80, edgecolors="k")
    plt.xlabel("Tinggi (cm)")
    plt.ylabel("Berat (kg)")
    plt.title(f"Clustering Data Siswa (k={n_clusters})")
    cluster_plot = plot_to_base64()
    plt.close()

    # Kirim data ke template
    return render_template(
        "index.html",
        elbow_plot=elbow_plot,
        silhouette_scores=silhouette_scores,
        n_clusters=n_clusters,
        cluster_plot=cluster_plot,
        df=df.to_dict(orient="records")
    )

if __name__ == "__main__":
    app.run(debug=True)
