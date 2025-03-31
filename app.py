from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os
from sklearn.cluster import KMeans

# Flask uygulaması
app = Flask(__name__)

# ❗ Sadece batuhandurmaz.com'dan gelen istekler kabul edilir
CORS(app, origins=["https://www.batuhandurmaz.com"])

# OpenAI API key (Railway'de ortam değişkeni olarak tanımlanmalı)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Embedding alma fonksiyonu
def get_embeddings(keywords):
    response = openai.embeddings.create(
        input=keywords,
        model="text-embedding-ada-002"
    )
    return [item.embedding for item in response.data]

# Kümeleme (clustering) işlemi
def cluster_keywords(keywords, n_clusters=5):
    embeddings = get_embeddings(keywords)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(str(label), []).append(keywords[i])
    return clusters

# API endpoint
@app.route("/api/cluster-keywords", methods=["POST"])
def cluster():
    data = request.json
    keywords = data.get("keywords", [])
    num_clusters = data.get("clusters", 5)

    if not keywords or len(keywords) < 2:
        return jsonify({"error": "En az 2 anahtar kelime gönderilmelidir."}), 400

    try:
        result = cluster_keywords(keywords, n_clusters=num_clusters)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Railway için dinamik port ayarı
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
