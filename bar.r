import pandas as pd
import plotly.express as px

# -------------------------------------------------------
# 1. Data lezen
# -------------------------------------------------------
df = pd.read_excel("data/schoon.xlsx")   # gebruik opschone dataset als je die hebt

# fallback: als 'branche' soms lowercase of anders is
df["branche"] = df["branche"].astype(str).str.strip()

# -------------------------------------------------------
# 2. Tellen hoe vaak elke branche voorkomt
# -------------------------------------------------------
branch_counts = df["branche"].value_counts().reset_index()
branch_counts.columns = ["Branche", "Aantal"]

# -------------------------------------------------------
# 3. Bar chart
# -------------------------------------------------------
fig = px.bar(
    branch_counts,
    x="Branche",
    y="Aantal",
    title="Aantal deelnemers per branche",
    text="Aantal",
    color="Branche",
    color_discrete_sequence=px.colors.qualitative.Set2,  # zachte pastelkleuren
)

fig.update_traces(
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Aantal: %{y}<extra></extra>"
)

fig.update_layout(
    width=900,
    height=600,
    xaxis_title="Branche",
    yaxis_title="Aantal Deelnemers",
    showlegend=False,
    margin=dict(l=60, r=40, t=80, b=120)
)

# -------------------------------------------------------
# 4. Export
# -------------------------------------------------------
fig.write_html("plots/branch_barchart.html")
print("✔ Branch bar chart opgeslagen als plots/branch_barchart.html")
