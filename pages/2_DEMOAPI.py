import streamlit as st
import pandas as pd
import numpy as np
import pickle
# import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import json
import plotly.express as px
import altair as alt
import seaborn as sns
import requests
import plotly.graph_objects as go



def identifiant_client():
    SK_ID_CURR = st.sidebar.selectbox('SK_ID_CURR', (X.SK_ID_CURR))
    data = {'SK_ID_CURR': SK_ID_CURR}
    ID_client = pd.DataFrame(data, index=[0])
    return data , ID_client


#if __name__ == "__main__":
st.set_page_config(
    page_title="Streamlit basics app",
    layout="centered"
)

st.title("Application qui prédit l'accord du crédit")

st.write("Auteur : Alexandre Jacqueline  - Data Scientist")

# Display the LOGO
#    img = Image.open("LOGO.png")
#    st.sidebar.image(img, width=300)

# Collecter le profil d'entrée
st.sidebar.header("Identifiant du client")

X = pd.read_csv('x_test.csv')

# Variables sélectionnées
df_vars_selected = pd.read_csv('df_vars_selected_saved.csv')
vars_selected = df_vars_selected['col_name'].to_list()



# Afficher les données du client:
vars_selected.insert(0, 'SK_ID_CURR')  # Ajout de l'identifiant aux features
st.subheader('1. Les données du client')

data, input_df = identifiant_client() #.iloc[0, 0]
input_df = input_df.iloc[0, 0]
X = X[vars_selected]
donnees_client = X[X['SK_ID_CURR'] == input_df]  # ligne du dataset qui concerne le client
st.write(donnees_client)



vars_selected.insert(0, 'SK_ID_CURR')  # Ajout de l'identifiant aux features
st.subheader('1. Les données du client')
data = {"SK_ID_CURR": float(input_df)}
API_ENDPOINT = "https://powerful-tor-12957.herokuapp.com/predict"


#data = {"SK_ID_CURR": float(input_df)}

#st.write(data)
# sending post request and saving response as response object

r = requests.post(url=API_ENDPOINT, json=data).json()
#st.write(r)
classe = r["classe"]
prevision = r['prevision']

st.write(prevision)
st.write(classe)

st.markdown(""" <br> <br> """, unsafe_allow_html=True)
st.write(f"Le client dont l'identifiant est **{data}** a obtenu le score de **{round(prevision*100,2):}%**.") #**{prevision:.1f}%**.")
st.write(f"**Il y a donc un risque de {prevision:.1f}% que le client ait des difficultés de paiement.**")
st.write(f"Le client est donc considéré par *'Prêt à dépenser'* comme **{classe}** \
        et décide de lui **{classe}** le crédit. ")

col1, col2 = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant

# Impression du graphique jauge
with col1:
    fig = go.Figure(go.Indicator(
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    value = round(prevision*100,3),          #float(prevision),
                    mode = "gauge+number+delta",
                    title = {'text': "Score du client", 'font': {'size': 24}},
                    delta = {'reference': 35.2, 'increasing': {'color': "#3b203e"}},
                    gauge = {'axis': {'range': [None, 100],
                            'tickwidth': 3,
                            'tickcolor': 'darkblue'},
                            'bar': {'color': 'white', 'thickness' : 0.3},
                            'bgcolor': 'white',
                            'borderwidth': 1,
                            'bordercolor': 'gray',
                            'steps': [{'range': [0, 20], 'color': '#e8af92'},
                                    {'range': [20, 40], 'color': '#db6e59'},
                                    {'range': [40, 60], 'color': '#b43058'},
                                    {'range': [60, 80], 'color': '#772b58'},
                                    {'range': [80, 100], 'color': '#3b203e'}],
                            'threshold': {'line': {'color': 'white', 'width': 8},
                                        'thickness': 0.8,
                                        'value': 35.2 }}))

fig.update_layout(paper_bgcolor='white',
                height=400, width=500,
                font={'color': '#772b58', 'family': 'Roboto Condensed'},
                margin=dict(l=30, r=30, b=5, t=5))
st.plotly_chart(fig, use_container_width=True)

#prevision = json.loads(prevision)

# st.write(prevision["reponse"])
# st.write(type(prevision))

# Appliquer le modèle sur le profil d'entrée
# Calcul des valeurs Shap
