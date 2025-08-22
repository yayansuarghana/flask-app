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

    # Grafik Elbow
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

    # Grafik clustering
    plt.figure(figsize=(5, 3))
    plt.scatter(df["Tinggi"], df["Berat"], c=df["Cluster"], cmap="viridis", s=80, edgecolors="k")
    plt.xlabel("Tinggi (cm)")
    plt.ylabel("Berat (kg)")
    plt.title(f"Clustering Data Siswa (k={n_clusters})")
    cluster_plot = plot_to_base64()
    plt.close()

    # Hitung centroid / rata-rata tiap cluster
    centroids = df.groupby("Cluster")[["Tinggi", "Berat"]].mean().round(2).reset_index()
    centroid_list = centroids.to_dict(orient="records")

    # Grafik distribusi jumlah siswa per cluster
    plt.figure(figsize=(5,3))
    df["Cluster"].value_counts().sort_index().plot(kind="bar", color=["skyblue","orange","green","red","purple","brown","gray"])
    plt.title("Distribusi Jumlah Siswa per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Jumlah Siswa")
    distribusi_plot = plot_to_base64()
    plt.close()

    # Grafik rata-rata per cluster
    plt.figure(figsize=(6,4))
    centroids.plot(x="Cluster", kind="bar", figsize=(6,4))
    plt.title("Profil Rata-rata per Cluster")
    plt.ylabel("Rata-rata")
    profil_plot = plot_to_base64()
    plt.close()

    # Kirim ke template
    return render_template(
        "index-3.html",
        elbow_plot=elbow_plot,
        silhouette_scores=silhouette_scores,
        n_clusters=n_clusters,
        cluster_plot=cluster_plot,
        df=df.to_dict(orient="records"),
        centroids=centroid_list,
        distribusi_plot=distribusi_plot,
        profil_plot=profil_plot
    )

if __name__ == "__main__":
    app.run(debug=True)
