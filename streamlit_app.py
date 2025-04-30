import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

# -------------------------------
# Chargement des données
# -------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv("data/data.csv", parse_dates=["Date"])
    preds = {
        "Naïf": pd.read_csv("data/preds_naive.csv", parse_dates=["Date"]),
        "Exponential Smoothing": pd.read_csv("data/preds_expsmoothing.csv", parse_dates=["Date"]),
        "Holt-Winters": pd.read_csv("data/preds_holtwinters.csv", parse_dates=["Date"]),
        "ETSformer": pd.read_csv("data/preds_etsformer.csv", parse_dates=["Date"]),
        "LGBMxProphet": pd.read_csv("data/preds_lgbmxprophet.csv", parse_dates=["Date"]),
    }
    return data, preds

df, preds_dict = load_data()

# -------------------------------
# Titre et introduction
# -------------------------------

st.set_page_config(page_title="Dashboard prévision ventes", layout="wide")
st.title("Dashboard - Prévision des ventes vs Actuals")
st.markdown("""
Ce tableau de bord permet de visualiser les ventes historiques par magasin,
et de comparer les performances de plusieurs modèles de prévision.
""")

# -------------------------------
# Analyse exploratoire des données
# -------------------------------

st.header("Analyse exploratoire")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe(), use_container_width=True)

with col2:
    st.subheader("Nombre de jours ouverts / fermés")
    opened_counts = df["Open"].value_counts().rename({1: "Ouvert", 0: "Fermé"})
    st.bar_chart(opened_counts)

# Graphique interactif : ventes moyennes par magasin
st.subheader("Ventes moyennes par magasin")
avg_sales = df.groupby("Store")["Sales"].mean().reset_index()
fig1 = px.bar(avg_sales, x="Store", y="Sales", labels={"Sales": "Ventes moyennes"})
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# Sélection du magasin
# -------------------------------

st.header("Sélection du magasin")
store_list = sorted(df["Store"].unique())
selected_store = st.selectbox("Sélectionnez un magasin :", store_list)

store_data = df[df["Store"] == selected_store]

st.line_chart(store_data.set_index("Date")["Sales"], height=300)

# -------------------------------
# Comparaison des modèles
# -------------------------------

st.header("Comparaison des modèles de prévision")
available_models = list(preds_dict.keys())

selected_models = st.multiselect(
    "Choisissez les modèles à afficher :",
    available_models,
    default=["LGBMxProphet"]
)

fig2 = go.Figure()

for model_name in selected_models:
    model_preds = preds_dict[model_name]
    store_preds = model_preds[model_preds["Store"] == selected_store]
    fig2.add_trace(go.Scatter(
        x=store_preds["Date"],
        y=store_preds["Sales"],
        mode="lines",
        name=model_name
    ))

# Ajout de la vraie série pour comparaison
fig2.add_trace(go.Scatter(
    x=store_data["Date"],
    y=store_data["Sales"],
    mode="lines",
    name="Ventes réelles",
    line=dict(color="black", dash="dot")
))

fig2.update_layout(
    title=f"Prévisions vs Réel - Magasin {selected_store}",
    xaxis_title="Date",
    yaxis_title="Ventes",
    legend_title="Modèle",
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Accessibilité (texte alternatif / descriptif)
# -------------------------------

st.markdown("""
---
ℹ**Accessibilité :** les graphiques utilisent un contraste fort, des légendes explicites, 
et les informations sont également accessibles en texte brut pour les lecteurs d'écran.
""")
