import mysql.connector
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

def run_kmeans():
    # ===== Ambil data dari database =====
    db = mysql.connector.connect(
        host="localhost",
        user="root",         # ganti username MySQL
        password="",         # ganti password MySQL
        database="testdb"    # ganti nama database
    )
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, nama, tinggi, berat FROM data_siswa")
    rows = cursor.fetchall()

    if not rows:
        return [], None

    X = [(r['tinggi'], r['berat']) for r in rows]

    # ===== KMeans =====
    k = 2  # awal coba 2 cluster
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Tambahkan cluster ke data
    for i, r in enumerate(rows):
        r['cluster'] = int(labels[i])

    # ===== Visualisasi =====
    plt.figure(figsize=(6,5))
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
    for i, r in enumerate(rows):
        plt.scatter(r['tinggi'], r['berat'], color=colors[r['cluster'] % len(colors)], s=80)
    plt.xlabel("Tinggi")
    plt.ylabel("Berat")
    plt.title(f"KMeans Clustering (k={k})")

    # Simpan plot
    plot_path = "static/cluster_plot.png"
    os.makedirs("static", exist_ok=True)
    plt.savefig(plot_path)
    plt.close()

    return rows, plot_path
