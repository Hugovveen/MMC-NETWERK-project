import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

# -------------------------------------------------------
# 1. Data lezen
# -------------------------------------------------------
df = pd.read_excel("data/schoon.xlsx")   # gebruik je opgeschoonde dataset!

# Skills naar string
df["clean_skills"] = df["clean_skills"].astype(str)

# -------------------------------------------------------
# 2. Vectorization – clean lijstjes flattenen
# -------------------------------------------------------
vectorizer = CountVectorizer(token_pattern=r"[^',\[\] ]+")
X = vectorizer.fit_transform(df["clean_skills"])

skill_names = vectorizer.get_feature_names_out()

# DataFrame: deelnemers × skills
skill_df = pd.DataFrame(X.toarray(), columns=skill_names)
skill_df["name"] = df["First Name"] + " " + df["Last Name"]

# -------------------------------------------------------
# 3. Filter op top N skills (belangrijk!)
# -------------------------------------------------------
top_n = 15   # verander naar 15–30 afhankelijk van wat het mooiste is

# Tel hoeveel keer elke skill voorkomt
skill_counts = skill_df.drop(columns=["name"]).sum().sort_values(ascending=False)

# De top N meest voorkomende skills
top_skills = skill_counts.head(top_n).index.tolist()

# Filter voor heatmap
filtered = skill_df[top_skills]
filtered.index = skill_df["name"]

# -------------------------------------------------------
# 4. Heatmap genereren
# -------------------------------------------------------
fig = px.imshow(
    filtered.values,
    labels=dict(x="Skill", y="Deelnemer", color="Aanwezigheid"),
    x=filtered.columns,
    y=filtered.index,
    aspect="auto",
    color_continuous_scale="Blues"
)

# -------------------------------------------------------
# 5. Layout pimpen
# -------------------------------------------------------
fig.update_layout(
    title=f"Skill Heatmap – Top {top_n} Skills (Opgeschoond)",
    width=1400,
    height=900,
    xaxis=dict(tickangle=45),          # gedraaide labels
    margin=dict(l=120, r=50, t=100, b=150)
)

# -------------------------------------------------------
# 6. Export
# -------------------------------------------------------
fig.write_html("plots/heatmap.html")
print("✔ Heatmap opgeslagen als plots/heatmap.html")
