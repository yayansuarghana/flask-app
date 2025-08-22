from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")  # penting: backend non-GUI agar tidak error di thread/non-display
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# ---- Helper: convert plot ke base64 ----
def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

@app.route("/", methods=["GET", "POST"])
def index():
    # 1) Baca data
    df = pd.read_csv("data.csv")

    # Pastikan kolom numerik valid
    for col in ["Tinggi", "Berat"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Tinggi", "Berat"]).reset_index(drop=True)

    X = df[["Tinggi", "Berat"]]

    # 2) Elbow & Silhouette (k = 2..7)
    inertias = []
    sil_scores = []
    K = range(2, 8)
    for k in K:
        km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_tmp = km_tmp.fit_predict(X)
        inertias.append(km_tmp.inertia_)
        sil = silhouette_score(X, labels_tmp)
        sil_scores.append((k, round(sil, 3)))

    # Elbow plot
    plt.figure(figsize=(5, 3))
    plt.plot(list(K), inertias, marker="o")
    plt.title("Metode Elbow")
    plt.xlabel("Jumlah Cluster (k)")
    plt.ylabel("Inertia")
    elbow_plot = plot_to_base64()
    plt.close()

    # 3) Ambil pilihan cluster dari dropdown
    try:
        n_clusters = int(request.form.get("n_clusters", 3))
    except:
        n_clusters = 3
    n_clusters = max(2, min(7, n_clusters))  # jaga-jaga

    # 4) Fit KMeans sesuai pilihan
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X)

    # 5) Scatter clustering + centroid
    plt.figure(figsize=(5, 3))
    plt.scatter(df["Tinggi"], df["Berat"], c=df["Cluster"], cmap="viridis", s=80, edgecolors="k", alpha=0.8)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c="red", s=220, marker="X", label="Centroid")
    for i, (cx, cy) in enumerate(centers):
        plt.text(cx, cy, f"  C{i}", va="center", fontsize=9)
    plt.xlabel("Tinggi (cm)")
    plt.ylabel("Berat (kg)")
    plt.title(f"Clustering Data Siswa (k={n_clusters})")
    plt.legend()
    cluster_plot = plot_to_base64()
    plt.close()

    # 6) Centroid table (rata-rata)
    centroids = []
    for i, (cx, cy) in enumerate(centers):
        centroids.append({
            "Cluster": i,
            "Rata_Tinggi": round(float(cx), 2),
            "Rata_Berat": round(float(cy), 2),
        })

    # 7) Ringkasan statistik per cluster
    summary_df = (
        df.groupby("Cluster")
          .agg(
              Jumlah=("Nama", "count"),
              Min_Tinggi=("Tinggi", "min"),
              Max_Tinggi=("Tinggi", "max"),
              Mean_Tinggi=("Tinggi", "mean"),
              Std_Tinggi=("Tinggi", "std"),
              Min_Berat=("Berat", "min"),
              Max_Berat=("Berat", "max"),
              Mean_Berat=("Berat", "mean"),
              Std_Berat=("Berat", "std"),
          )
          .reset_index()
    )
    summary = summary_df.to_dict(orient="records")

    # 8) Distribusi anggota (pie)
    plt.figure(figsize=(4, 4))
    df["Cluster"].value_counts().sort_index().plot.pie(autopct="%1.1f%%", startangle=90)
    plt.ylabel("")
    plt.title("Distribusi Jumlah Siswa per Cluster")
    distribusi_plot = plot_to_base64()
    plt.close()

    # 9) Boxplot tinggi & berat per cluster
    plt.figure(figsize=(6, 4))
    df.boxplot(column="Tinggi", by="Cluster", grid=False)
    plt.title("Boxplot Tinggi per Cluster")
    plt.suptitle("")
    tinggi_boxplot = plot_to_base64()
    plt.close()

    plt.figure(figsize=(6, 4))
    df.boxplot(column="Berat", by="Cluster", grid=False)
    plt.title("Boxplot Berat per Cluster")
    plt.suptitle("")
    berat_boxplot = plot_to_base64()
    plt.close()

    # 10) Data per cluster (untuk tabel per cluster di halaman)
    clusters = {}
    for cid in sorted(df["Cluster"].unique()):
        clusters[int(cid)] = df[df["Cluster"] == cid].to_dict(orient="records")

    # 11) Simpan hasil clustering (untuk download)
    df.to_csv("hasil_clustering.csv", index=False)

    # 12) Render
    return render_template(
        "index.html",
        elbow_plot=elbow_plot,
        silhouette_scores=sil_scores,
        n_clusters=n_clusters,
        cluster_plot=cluster_plot,
        distribusi_plot=distribusi_plot,
        tinggi_boxplot=tinggi_boxplot,
        berat_boxplot=berat_boxplot,
        centroids=centroids,
        df=df.to_dict(orient="records"),
        summary=summary,
        clusters=clusters,
    )

@app.route("/download")
def download():
    return send_file("hasil_clustering.csv", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
