#imports
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#READ IN
df = pd.read_excel("data/Deelnemers 18-11.xlsx")

df["text"] = (
    df["functie"].fillna("") + " " + 
    df["huidig werk"].fillna("") + " " +
    df["skills"].fillna("")
)

#TF-IDF
vectorizer = TfidfVectorizer(stop_words=None, max_features=300)
X = vectorizer.fit_transform(df["text"])

#DIMENSIONS (PCA NAAR 2D)
pca = PCA(n_components=2)
coords = pca.fit_transform(X.toarray())

df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

#ClUSTERING (KMeans)
k = 4
kmeans = KMeans(n_clusters=k,random_state=42)
df["cluster"] = kmeans.fit_predict(coords)

#PLOT

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="cluster",
    hover_name="First Name",
    hover_data=["Last Name", "functie", "huidig werk", "skills"],
    title="Clusterplot van Deelnemers (Plotly Interactief)",
    width=900,
    height=600
)

fig.update_traces(marker=dict(size=12, opacity=0.8))

fig.write_html("plots/cluster_scatter.html")
print("Plot opgeslagen als plots/cluster_scatter.html")

