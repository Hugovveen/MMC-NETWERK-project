import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# -------------------------------------------------------
# 1. Load cleaned data
# -------------------------------------------------------
df = pd.read_excel("data/schoon.xlsx")

# Combine tekstvelden exact zoals in jouw main.py
df["text"] = (
    df["functie"].fillna("") + " " +
    df["huidig werk"].fillna("") + " " +
    df["clean_skills"].astype(str)
)

# -------------------------------------------------------
# 2. TF-IDF vectorization
# -------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words=None, max_features=300)
X = vectorizer.fit_transform(df["text"])

# -------------------------------------------------------
# 3. PCA to 2D (same reduction as your plot)
# -------------------------------------------------------
pca = PCA(n_components=2)
coords = pca.fit_transform(X.toarray())

df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

# -------------------------------------------------------
# 4. Re-run KMeans (same settings as main.py)
# -------------------------------------------------------
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster"] = kmeans.fit_predict(coords)

# -------------------------------------------------------
# 5. Print useful cluster overview
# -------------------------------------------------------
clusters = sorted(df["cluster"].unique())

for c in clusters:
    print("\n" + "="*70)
    print(f" CLUSTER {c} — VOORLOPIG OVERZICHT")
    print("="*70)

    subset = df[df["cluster"] == c]

    for _, row in subset.iterrows():
        name = f"{row['First Name']} {row['Last Name']}".strip()
        functie = str(row["functie"])
        werk = str(row["huidig werk"])

        # clean_skills is a Python list in string form → eval fix
        try:
            skills_list = eval(row["clean_skills"])
        except:
            skills_list = row["clean_skills"]
        skills = ", ".join(skills_list)

        print(f"""
Naam:         {name}
Functie:      {functie}
Huidig werk:  {werk}
Skills:       {skills}
----------------------------------------------------""")
