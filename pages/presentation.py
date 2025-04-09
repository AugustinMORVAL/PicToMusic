import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import os

# Configuration de la page
st.set_page_config(
    page_title="SonataBene - Présentation",
    page_icon="🎵",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .big-font {
        font-size: 2.5em !important;
        font-weight: bold;
        color: #4B8BBE;
    }
    .medium-font {
        font-size: 1.5em !important;
        color: #306998;
    }
    .highlight {
        background-color: #FFE873;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .section {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre et slogan
st.markdown('<p class="big-font">SonataBene</p>', unsafe_allow_html=True)
st.markdown('<p class="medium-font">Transformer vos images en mélodies</p>', unsafe_allow_html=True)

# Problématique
st.markdown("## 🎯 Problématique")
st.markdown("""
<div class="highlight">
Comment transformer simplement une image en mélodie ?
</div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("## 🎼 Introduction")
st.markdown("""
<div class="section">
SonataBene c'est quoi? Une application qui transforme les partitions musicales en fichiers MIDI jouables. \n
Elle combine des techniques de traitement d'image traditionnelles et des modèles d'apprentissage profond pour une reconnaitre les notations musicales.
</div>
""", unsafe_allow_html=True)

# Fonctionnalités
st.markdown("## ⚙️ Fonctionnalités")
st.markdown("""
<div class="section">
            
### 🎼 Reconnaissance Musicale
<div style='display: flex; justify-content: space-between; margin: 20px 0;'>
    <div style='width: 45%; padding: 15px; background-color: #4B8BBE; color: white; border-radius: 10px;'>
        <h4>🎵 Notes</h4>
        <ul style='color: white;'>
            <li>Détection des notes</li>
            <li>Reconnaissance de la hauteur</li>
            <li>Détermination de la durée</li>
        </ul>
    </div>
    <div style='width: 45%; padding: 15px; background-color: #306998; color: white; border-radius: 10px;'>
        <h4>🎶 Symboles</h4>
        <ul style='color: white;'>
            <li>Détection des clés</li>
            <li>Reconnaissance des armatures</li>
            <li>Identification des silences</li>
        </ul>
    </div>
</div>

### 🔄 Conversion et Génération
<div style='display: flex; justify-content: space-between; margin: 20px 0;'>
    <div style='width: 30%; padding: 15px; background-color: #f8f9fa; border-radius: 10px;'>
        <h4>📝 Transcription en format musical</h4>
        <ul>
            <li>Notation textuelle standard</li>
            <li>Facile à éditer et partager</li>
            <li>Compatible avec de nombreux outils</li>
        </ul>
    </div>
    <div style='width: 30%; padding: 15px; background-color: #f8f9fa; border-radius: 10px;'>
        <h4>🎵 Génération Audio</h4>
        <ul>
            <li>Support multi-instruments</li>
            <li>Contrôle du tempo et de la dynamique</li>
            <li>Export en différents formats audio</li>
        </ul>
    </div>
    <div style='width: 30%; padding: 15px; background-color: #f8f9fa; border-radius: 10px;'>
        <h4>🎼 Export MuseScore</h4>
        <ul>
            <li>Édition interactive des partitions</li>
            <li>Modification en temps réel</li>
            <li>Export vers différents formats</li>
        </ul>
    </div>
</div>
</div>
""", unsafe_allow_html=True)

# Cas d'usage
st.markdown("## 💡 Cas d'usage")
st.markdown("""
<div class="section">
<div style='display: flex; justify-content: space-between; margin: 20px 0;'>
    <div style='width: 45%; padding: 15px; background-color: #4B8BBE; color: white; border-radius: 10px;'>
        <h4>🎼 Composition Assistée</h4>
        <ul style='color: white;'>
            <li>Capture rapide d'idées musicales</li>
            <li>Retranscription instantanée en partition</li>
            <li>Export direct vers MuseScore pour l'édition</li>
        </ul>
    </div>
    <div style='width: 45%; padding: 15px; background-color: #306998; color: white; border-radius: 10px;'>
        <h4>🏛️ Patrimoine culturel</h4>
        <ul style='color: white;'>
            <li>Reconstruction de partitions anciennes</li>
            <li>Préservation du patrimoine musical</li>
            <li>Numérisation de documents historiques</li>
        </ul>
    </div>
</div>

<div style='display: flex; justify-content: space-between; margin: 20px 0;'>
    <div style='width: 45%; padding: 15px; background-color: #FFE873; color: black; border-radius: 10px;'>
        <h4>🎓 Pédagogie</h4>
        <ul>
            <li>Apprentissage assisté</li>
            <li>Possibilité d'écouter la partition</li>
            <li>Visualisation interactive</li>
        </ul>
    </div>
    <div style='width: 45%; padding: 15px; background-color: #FFB74D; color: black; border-radius: 10px;'>
        <h4>🎹 Accompagnement musical</h4>
        <ul>
            <li>Aide à la création de partitions</li>
            <li>Retranscription automatique</li>
            <li>Support pour débutants</li>
        </ul>
    </div>
</div>
</div>
""", unsafe_allow_html=True)

# Pipeline
st.markdown("## 🔄 Notre Pipeline")
st.markdown("""
<div class="section">
### 🎯 Objectif
Construire un dataset de qualité pour entraîner notre modèle de reconnaissance musicale
</div>
""", unsafe_allow_html=True)


# Pipeline visuel avec étapes
st.markdown("### 🔄 Processus de Construction du Dataset")

# Création d'un pipeline visuel avec des cartes
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #4B8BBE; color: white; border-radius: 10px;'>
    <h3>📚 Collection</h3>
    <p>87,678 paires</p>
    <p>PNG/XML</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #306998; color: white; border-radius: 10px;'>
    <h3>✂️ Découpe</h3>
    <p>87,678 images</p>
    <p>OpenCV</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #FFE873; color: black; border-radius: 10px;'>
    <h3>🏷️ Labellisation</h3>
    <p>87,678 xml</p>
    <p>XML → ABC</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #FFB74D; color: black; border-radius: 10px;'>
    <h3>🎵 Dataset Final</h3>
    <p>49,000 images annotées</p>
    <p>Prêt pour l'entraînement</p>
    </div>
    """, unsafe_allow_html=True)

# Détails du processus
st.markdown("""
<div class="section">
### 📝 Détails du Processus

<div style='display: flex; justify-content: space-between; margin: 20px 0;'>
    <div style='width: 30%; padding: 15px; background-color: #f8f9fa; border-radius: 10px;'>
        <h4>🎯 Collection Initiale</h4>
        <ul>
            <li>Dataset PrIMuS</li>
            <li>Format standardisé</li>
            <li>87,678 paires</li>
        </ul>
    </div>
    <div style='width: 30%; padding: 15px; background-color: #f8f9fa; border-radius: 10px;'>
        <h4>🔧 Prétraitement</h4>
        <ul>
            <li>Découpe OpenCV</li>
            <li>250 images test</li>
            <li>Vérification manuelle</li>
        </ul>
    </div>
    <div style='width: 30%; padding: 15px; background-color: #f8f9fa; border-radius: 10px;'>
        <h4>🎼 Finalisation</h4>
        <ul>
            <li>Traduction XML/ABC</li>
            <li>Matching notes/labels</li>
            <li>60% annoté</li>
        </ul>
    </div>
</div>
</div>
""", unsafe_allow_html=True)

# Démo
st.markdown("## 🎮 Démo")

# Création d'un bouton pour accéder à la démo
if st.button("🚀 Accéder à la démo End2End Pipeline", type="primary"):
    st.switch_page("./0_End2End_Pipeline.py")

st.markdown("</div>", unsafe_allow_html=True)

# Key Learnings
st.markdown("## 📚 Retours d'Expérience & Perspectives")
st.markdown("""
<div class="section">

### 🎯 Points d'Amélioration
- **Modèle d'Apprentissage**
  - Besoin d'un entraînement plus approfondi
  - Amélioration de la précision sur les notes complexes
  - Support des partitions multi-lignes (mains droite/gauche)

- **Architecture**
  - Approche modulaire séparant rythme et notes
  - Optimisation du pipeline de traitement
  - Meilleure gestion des cas particuliers

### 🚀 Perspectives
- Extension du dataset avec des partitions plus variées
- Intégration de l'analyse harmonique
- Support des partitions orchestrales
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>SonataBene - Transformer vos images en mélodies</p>
</div>
""", unsafe_allow_html=True) 