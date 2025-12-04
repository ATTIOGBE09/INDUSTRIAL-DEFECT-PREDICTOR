import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Défaut Production - Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chargement des données et modèle
df = pd.read_csv("dataset_defaut.csv")
model = joblib.load("modele_defaut.pkl")

features = ['ProductionVolume', 'ProductionCost', 'EnergyConsumption',
            'QualityScore', 'MaintenanceHours']

# Sidebar (Menu latéral)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/8291/8291359.png", width=80)
st.sidebar.title(" Menu principal")
section = st.sidebar.radio("Aller vers :", ["Exploration", "Prédiction", "Recommandations"])

# En-tête
st.markdown("<h1 style='text-align:center;color:#3E64FF;'> Analyse intelligente des défauts de production</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #ccc'>", unsafe_allow_html=True)

# SECTION 1 : Exploration
if section == "Exploration":
    st.subheader(" Visualisation des données par variable")
    col1, col2 = st.columns([1, 3])
    with col1:
        variable = st.selectbox("Choisissez une variable :", features)

    with col2:
        fig = px.box(df, x="DefectStatus", y=variable, color="DefectStatus",
                     points="outliers",
                     labels={"DefectStatus": "Défaut"},
                     color_discrete_map={0: "#2ca02c", 1: "#d62728"},
                     title=f"{variable} selon le statut de défaut")
        fig.update_layout(
            xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Sans défaut", "Avec défaut"]),
            showlegend=False, title_x=0.5
        )
        st.plotly_chart(fig, use_container_width=True)

    # Statut de défaut global
    st.markdown("###  Distribution globale des jours avec défaut")
    count_df = df["DefectStatus"].value_counts().reset_index()
    count_df.columns = ["Statut", "Jours"]
    count_df["Statut"] = count_df["Statut"].map({0: "Sans défaut", 1: "Avec défaut"})
    st.dataframe(count_df)

# SECTION 2 : Prédiction
elif section == "Prédiction":
    st.subheader(" Prédiction en temps réel")

    st.markdown("Remplissez les champs ci-dessous pour simuler une situation de production.")

    cols = st.columns(3)
    user_input = {}
    for i, feature in enumerate(features):
        user_input[feature] = cols[i % 3].number_input(
            label=feature,
            value=float(np.median(df[feature])),
            format="%.2f"
        )

    if st.button(" Prédire le défaut probable"):
        input_df = pd.DataFrame([user_input])
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(f" Défaut probable détecté ! (probabilité : {prob:.1%})")
        else:
            st.success(f" Aucun défaut détecté (probabilité : {prob:.1%})")

# SECTION 3 : Recommandations
elif section == "Recommandations":
    st.subheader(" Recommandations intelligentes")

    st.markdown("Voici des recommandations générées automatiquement à partir de vos données récentes.")

    latest_entry = df[features].iloc[-1:]  # dernier jour de prod

    quality = latest_entry["QualityScore"].values[0]
    maintenance = latest_entry["MaintenanceHours"].values[0]
    energy = latest_entry["EnergyConsumption"].values[0]
    volume = latest_entry["ProductionVolume"].values[0]

    if quality < 80:
        st.warning(" Qualité produit faible : améliorer le **QualityScore**.")
    else:
        st.success(" Qualité produit satisfaisante.")

    if maintenance > 12:
        st.warning(" Trop d’heures de maintenance : planifier des maintenances ciblées.")
    else:
        st.success(" Niveau de maintenance optimal.")

    if energy > 4000:
        st.warning(" Consommation d’énergie élevée : ajuster les équipements.")
    else:
        st.success(" Consommation énergétique sous contrôle.")

    if volume > 800:
        st.warning(" Volume de production élevé : surveiller les surcharges.")
    else:
        st.success(" Volume de production raisonnable.")
