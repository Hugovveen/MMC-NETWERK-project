import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer

# Data lezen
df = pd.read_excel("data/Deelnemers 18-11.xlsx")

# Skills naar string
df["skills"] = df["skills"].fillna("").astype(str)

# Vectorize skilss
vectorizer = CountVectorizer(token_pattern=r"[^•,;]+")
X = vectorizer.fit_transform(df["skills"])

skill_names = vectorizer.get_feature_names_out()

# DataFrame: deelnemers × skills
skill_df = pd.DataFrame(X.toarray(), columns=skill_names)
skill_df["name"] = df["First Name"] + " " + df["Last Name"]

# Heatmap
fig = px.imshow(
    skill_df.drop(columns=["name"]).values,
    labels=dict(x="Skill", y="Deelnemer", color="Aanwezigheid"),
    x=skill_names,
    y=skill_df["name"].tolist(),
    aspect="auto",
    color_continuous_scale="Blues"
)

fig.update_layout(
    title="Skill Heatmap – Overlap tussen deelnemers",
    width=1200,
    height=900
)

fig.write_html("plots/heatmap.html")

print("Heatmap opgeslagen als plots/heatmap.html")