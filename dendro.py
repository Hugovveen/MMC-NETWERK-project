import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.figure_factory as ff

df = pd.read_excel("data/Deelnemers 18-11.xlsx")

df["text"] = (
    df["functie"].fillna("") + " " +
    df["huidig werk"].fillna("") + " " +
    df["skills"].fillna("")
)

vectorizer = TfidfVectorizer(max_features=300)
X = vectorizer.fit_transform(df["text"]).toarray()

dist_matrix = cosine_distances(X)

fig = ff.create_dendrogram(dist_matrix, labels=df["First Name"].tolist())
fig.update_layout(width=1000, height=600)
fig.write_html("plots/dendrogram.html")
