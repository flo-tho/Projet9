import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os

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
# Chargement des données
# -------------------------------

@st.cache_data
def load_data():

    base_url = "https://raw.githubusercontent.com/flo-tho/Projet9/master/previsions/"
    actuals = pd.read_csv(base_url + "df_simple_selected_stores.csv", parse_dates=["date"])
    preds = {
        "Naïf": pd.read_csv(base_url + "naive_predictions.csv", parse_dates=["date"]),
        "Exponential Smoothing": pd.read_csv(base_url + "ses_predictions.csv", parse_dates=["date"]),
        "Holt-Winters": pd.read_csv(base_url + "holt_winters_predictions.csv", parse_dates=["date"]),
        "ETSformer": pd.read_csv(base_url + "etsformer_predictions.csv", parse_dates=["date"]),
        "LGBMxProphet": pd.read_csv(base_url + "lgbm_preds.csv", parse_dates=["date"]),
        "LGBMxProphet avc feat. exogenes": pd.read_csv(base_url + "lgbm_exog_preds.csv", parse_dates=["date"]),
    }
    return actuals, preds




df, preds_dict = load_data()



# -------------------------------
# Analyse exploratoire des données
# -------------------------------

st.header("Analyse exploratoire")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Statistiques descriptives")
    st.dataframe(df.describe(), use_container_width=True)

with col2:
    with col2:
        st.subheader("Analyse des ventes par magasin")

        sales_analysis = df.groupby('Store').agg({
            'Sales': [
                ('Moyenne', 'mean'),
                ('Minimum', 'min'),
                ('Maximum', 'max'),
                ('Écart-type', 'std'),
                ('Nb jours à zero vente(closed)', lambda x: (x == 0).sum()),
            ]
        })

        # Aplatir les colonnes multi-index
        sales_analysis.columns = sales_analysis.columns.droplevel(0)
        sales_analysis = sales_analysis.reset_index()

        st.dataframe(sales_analysis)


st.subheader("Distribution des ventes par Store")

df_box = df.copy()
df_box["Magasin"] = "Store " + df_box["Store"].astype(str)  # créer une colonne lisible
df_box["Magasin"] = pd.Categorical(df_box["Magasin"])         # forcer le traitement en catégories

fig_box = px.box(df_box, x='Magasin', y='Sales',
                 title="Distribution des ventes par magasin",
                 labels={'Sales': 'Ventes', 'Magasin': 'Magasin'},
                 category_orders={"Magasin": sorted(df_box["Magasin"].unique())},
                 height=600)

fig_box.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig_box, use_container_width=True)

# -------------------------------
# Sélection de la période d'analyse
# -------------------------------

st.header("Période d'analyse")

min_date = df["date"].min()
max_date = df["date"].max()

# Options proposées
start_date_options = {
    f"{min_date.date()}": min_date,
    "2014-01-01": pd.Timestamp("2014-01-01"),
    "2015-01-01": pd.Timestamp("2015-01-01"),
}

selected_start_label = st.selectbox("Date de début :", list(start_date_options.keys()))
start_date = start_date_options[selected_start_label]

# -------------------------------
# Sélection du magasin
# -------------------------------

st.header("Sélection du magasin")
store_list = sorted(df["Store"].unique())
selected_store = st.selectbox("Sélectionnez un magasin :", store_list)

# Filtrage par magasin et date
store_data = df[(df["Store"] == selected_store) & (df["date"] >= start_date)]

fig_sales = go.Figure(data=[
    go.Scatter(x=store_data["date"], y=store_data["Sales"], mode='lines', name='Ventes')
])

fig_sales.update_layout(
    title=f"Évolution des ventes pour le magasin {selected_store}",
    xaxis_title="Date",
    yaxis_title="Ventes (Actuals)",
    height=500,
    showlegend=True
)

st.plotly_chart(fig_sales, use_container_width=True)

# -------------------------------
# Comparaison des modèles
# -------------------------------

st.header("Comparaison des modèles de prévision")
available_models = list(preds_dict.keys())

selected_models = st.multiselect(
    "Choisissez les modèles à afficher :",
    available_models,
    default=["Naïf"]
)

# Couleurs fixes pour chaque modèle (8 couleurs daltonisme-friendly)
MODEL_COLORS = {
    "LGBMxProphet": "#E69F00",   # orange
    "LGBMxProphet avc feat. exogenes": "#F0E442",  # jaune
    "Exponential Smoothing": "#56B4E9",      # bleu clair
    "Holt-Winters": "#009E73",           # vert foncé
    "Naive": "#0072B2",          # bleu foncé
    # "Prophet": "#D55E00",        # rouge brique
    "Naïf": "#CC79A7",         # rose
}

fig2 = go.Figure()

# Courbes de prévision
for model_name in selected_models:
    model_preds = preds_dict[model_name]
    store_preds = model_preds[
        (model_preds["Store"] == selected_store) &
        (model_preds["date"] >= start_date)
    ]
    fig2.add_trace(go.Scatter(
        x=store_preds["date"],
        y=store_preds["Sales_forecast"],
        mode="lines",
        name=model_name,
        line=dict(color=MODEL_COLORS.get(model_name, None))  # couleur fixe
    ))

# Courbe réelle
fig2.add_trace(go.Scatter(
    x=store_data["date"],
    y=store_data["Sales"],
    mode="lines",
    name="Actuals",
    line=dict(color="black", dash="dot")
))

fig2.update_layout(
    title=f"Prévisions vs Réel - Magasin {selected_store}",
    xaxis_title="Date",
    yaxis_title="Ventes",
    legend_title="Modèle",
    height=600
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Calcul des scores pour le magasin sélectionné
# -------------------------------

st.subheader("Scores d’erreur pour chaque modèle")

results = []

for model_name in selected_models:
    model_preds = preds_dict[model_name]
    store_preds = model_preds[model_preds["Store"] == selected_store]
    merged = store_preds.merge(store_data, on="date", suffixes=("_pred", "_actual"))

    y_true = merged["Sales_actual"]
    y_pred = merged["Sales_forecast"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    rmpse = np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + 1e-8)))) * 100

    results.append({
        "Modèle": model_name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "RMPSE (%)": round(rmpse, 2),
    })

score_df = pd.DataFrame(results).sort_values("RMSE")
st.dataframe(score_df)



# -------------------------------
# Accessibilité (texte alternatif / descriptif)
# -------------------------------

st.markdown("""
---
ℹ **Accessibilité :** les graphiques utilisent un contraste fort, des légendes explicites, 
et les informations sont également accessibles en texte brut pour les lecteurs d'écran.
""")
