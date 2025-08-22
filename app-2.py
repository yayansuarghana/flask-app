# app.py
# ====== Import ======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import io, base64

app = Flask(__name__)

# ====== Baca Data ======
CSV_PATH = "data.csv"
df = pd.read_csv(CSV_PATH)

df.columns = [c.strip() for c in df.columns]
expected_cols = ["Tinggi", "Berat"]
for col in expected_cols:
    if col not in df.columns:
        raise ValueError(f"Kolom {col} tidak ditemukan di CSV")

X = df[expected_cols].dropna().astype(float)

# ====== Cari range cluster ======
k_min, k_max = 2, min(10, len(X) - 1)
K_range = list(range(k_min, k_max + 1))

wcss, sil_scores = [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X, labels))

# ====== Helper untuk buat plot base64 ======
def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ====== Route utama ======
@app.route("/", methods=["GET", "POST"])
def index():
    # default cluster
    selected_k = int(request.form.get("n_cluster", 3))

    # Fit KMeans
    kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X)
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=expected_cols)

    # ===== Elbow plot =====
    plt.figure()
    plt.plot(K_range, wcss, marker="o")
    plt.xlabel("Jumlah Cluster (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    elbow_plot = fig_to_base64()
    plt.close()

    # ===== Scatter clustering =====
    plt.figure()
    plt.scatter(X["Tinggi"], X["Berat"], c=df["Cluster"], cmap="viridis", s=80, alpha=0.8, edgecolors="k")
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                c="red", marker="X", s=200, label="Centroid")
    plt.xlabel("Tinggi (cm)")
    plt.ylabel("Berat (kg)")
    plt.title(f"Hasil Clustering (k={selected_k})")
    plt.legend()
    cluster_plot = fig_to_base64()
    plt.close()

    # ===== Daftar anggota cluster =====
    cluster_groups = {}
    for cid, group in df.groupby("Cluster"):
        cluster_groups[cid] = group[["Nama", "Tinggi", "Berat"]].to_dict(orient="records")

    # ===== Rata-rata centroid per cluster =====
    cluster_profiles = df.groupby("Cluster")[expected_cols].mean().round(2).to_dict(orient="index")

    return render_template("index.html",
                           elbow_plot=elbow_plot,
                           cluster_plot=cluster_plot,
                           K_range=K_range,
                           selected_k=selected_k,
                           sil_scores=list(zip(K_range, sil_scores)),
                           cluster_groups=cluster_groups,
                           cluster_profiles=cluster_profiles)

if __name__ == "__main__":
    app.run(debug=True)
