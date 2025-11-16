# imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import numpy as np

# -------------------------------------------------------
# 1. READ IN
# -------------------------------------------------------
df = pd.read_excel("data/schoon.xlsx")   # gebruik opgekuiste dataset!

df["text"] = (
    df["functie"].fillna("") + " " +
    df["huidig werk"].fillna("") + " " +
    df["clean_skills"].astype(str)
)

# -------------------------------------------------------
# 2. TF-IDF
# -------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words=None, max_features=300)
X = vectorizer.fit_transform(df["text"])

# -------------------------------------------------------
# 3. DIMENSIONS (PCA NAAR 2D)
# -------------------------------------------------------
pca = PCA(n_components=2)
coords = pca.fit_transform(X.toarray())

df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

# -------------------------------------------------------
# 4. CLUSTERING (KMeans)
# -------------------------------------------------------
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster"] = kmeans.fit_predict(coords)

# -------------------------------------------------------
# 5. CLUSTER LABELS (jij mag ze aanpassen)
# -------------------------------------------------------
cluster_names = {
    0: "Menselijke Multitools / Creatief-Operationeel",
    1: "Interim Elite / Organisatorisch Leiderschap",
    2: "IT-Leiders & Architecten",
    3: "Publieke Teamleiders & Trainers",
}

df["cluster_label"] = df["cluster"].map(cluster_names)

# -------------------------------------------------------
# 6. BEGIN FIGUUR
# -------------------------------------------------------
fig = px.scatter(
    df,
    x="x",
    y="y",
    color="cluster_label",
    hover_name="First Name",
    hover_data=["Last Name", "functie", "huidig werk", "clean_skills"],
    title="Deelnemer linkedin overlap plot (Plotly Interactief)",
    width=1000,
    height=700
)

fig.update_traces(marker=dict(size=12, opacity=0.9))

# -------------------------------------------------------
# 7. ACHTERGRONDZONES PER CLUSTER
# -------------------------------------------------------
for c in df["cluster"].unique():
    pts = df[df["cluster"] == c][["x", "y"]].values

    if len(pts) >= 3:  # convex hull werkt alleen met >=3 punten
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]

        base_color = px.colors.qualitative.Set2[int(c)]  # bv 'rgb(141,160,203)'
        # maak er 'rgba(141,160,203,0.15)' van voor transparante fill
        fill_rgba = base_color.replace("rgb", "rgba").replace(")", ",0.15)")

        fig.add_trace(go.Scatter(
            x=hull_pts[:, 0],
            y=hull_pts[:, 1],
            fill="toself",
            mode="none",
            fillcolor=fill_rgba,
            hoverinfo="skip",
            showlegend=False
        ))

# -------------------------------------------------------
# 8. CLUSTER LABELS IN DE FIGUUR
# -------------------------------------------------------
for c in df["cluster"].unique():
    cx = df[df["cluster"] == c]["x"].mean()
    cy = df[df["cluster"] == c]["y"].mean()
    fig.add_annotation(
        x=cx,
        y=cy,
        text=cluster_names[c],
        showarrow=False,
        font=dict(size=18, color="black"),
        opacity=0.9
    )

# -------------------------------------------------------
# 9. ASLABELS
# -------------------------------------------------------
fig.update_layout(
    xaxis_title="Component 1 (PCA)",
    yaxis_title="Component 2 (PCA)",
    legend_title="Cluster",
    margin=dict(l=60, r=60, t=80, b=60)
)

# -------------------------------------------------------
# 10. EXPORT
# -------------------------------------------------------
fig.write_html("plots/cluster_scatter.html")
print("✔ Gepimpte clusterplot opgeslagen als plots/cluster_scatter.html")
