import pandas as pd
import plotly.express as px

# -------------------------------------------------------
# 1. Data lezen
# -------------------------------------------------------
df = pd.read_excel("data/schoon.xlsx")

# Branche kolom normaliseren
df["branche"] = df["branche"].astype(str).str.strip()

# -------------------------------------------------------
# 2. Lege inputs verwijderen
# -------------------------------------------------------
df = df[
    ~df["branche"].isin(["", " ", "nan", "NaN", "None", "NONE"])
]
df = df.dropna(subset=["branche"])

# -------------------------------------------------------
# 3. Tellen
# -------------------------------------------------------
branch_counts = df["branche"].value_counts().reset_index()
branch_counts.columns = ["Branche", "Aantal"]

# -------------------------------------------------------
# 4. Pie chart (donut)
# -------------------------------------------------------
fig = px.pie(
    branch_counts,
    names="Branche",
    values="Aantal",
    title="Verdeling van deelnemers per branche",
    hole=0.45,  # donut effect
    color="Branche",
    color_discrete_sequence=px.colors.qualitative.Set2,
)

fig.update_traces(
    textposition="inside",
    textinfo="label+percent",
    hovertemplate="<b>%{label}</b><br>Aantal: %{value}<extra></extra>"
)

fig.update_layout(
    width=700,
    height=700,
    margin=dict(l=40, r=40, t=80, b=40)
)

# -------------------------------------------------------
# 5. Export
# -------------------------------------------------------
fig.write_html("plots/branch_pie.html")
print("✔ Donut chart opgeslagen als plots/branch_pie.html")
