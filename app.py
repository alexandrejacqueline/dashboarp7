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

#Add
import streamlit.components.v1 as components
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve

import pickle
import shap



def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def identifiant_client():
    SK_ID_CURR = st.selectbox('SK_ID_CURR', (X.SK_ID_CURR)) #sidebar
    data = {'SK_ID_CURR': SK_ID_CURR}
    ID_client = pd.DataFrame(data, index=[0])
    return data , ID_client

def liste_variable():
    st.header("Quel est l'Impact de chaque caractéristique sur la prédiction ?")

    var = st.selectbox('variable_toplot',["EXT_SOURCE_2","NAME_FAMILY_STATUS_Married","EXT_SOURCE_3","DAYS_EMPLOYED"])
    fig = plt.figure(figsize=(12,4))

    ax1 = fig.add_subplot(121)
    shap.dependence_plot(var, shap_values[1], X_bis,
                         interaction_index=None, alpha = 0.5,
                         ax=ax1, show = False)
    ax1.title.set_text("Graphique de dépendance" )

    ax2 = fig.add_subplot(122)
    shap.dependence_plot(var, shap_values[1], X_bis,
                         interaction_index="auto", alpha = 0.5,
                         ax=ax2, show = False)
    ax2.title.set_text("Graphique de dépendance et intéraction" )

    #plt.tight_layout()
    st.pyplot()

X = pd.read_csv('x_test.csv')
X_bis = X.copy().drop(columns=["Unnamed: 0"])
model_LGBM = pickle.load(open("lightgbm.pkl", "rb"))


st.title("Loan application scoring dashboard")

st.write("Auteur : Alexandre Jacqueline  - Data Scientist")

# Collecter le profil d'entrée
st.header("Identifiant du client") #sidebar

# Variables sélectionnées
df_vars_selected = pd.read_csv('df_vars_selected_saved.csv')
vars_selected = df_vars_selected['col_name'].to_list()

# Afficher les données du client:
vars_selected.insert(0, 'SK_ID_CURR')  # Ajout de l'identifiant aux features
st.subheader('Les données du client')

data, input_df = identifiant_client() #.iloc[0, 0]
input_df = input_df.iloc[0, 0]
X = X[vars_selected]
donnees_client = X[X['SK_ID_CURR'] == input_df]  # ligne du dataset qui concerne le client
st.write(donnees_client)

vars_selected.insert(0, 'SK_ID_CURR')  # Ajout de l'identifiant aux features

data = {"SK_ID_CURR": float(input_df)}

#API
API_ENDPOINT = "https://powerful-tor-12957.herokuapp.com/predict"
# sending post request and saving response as response object
r = requests.post(url=API_ENDPOINT, json=data).json()

#data = {"SK_ID_CURR": float(input_df)}

#st.write(data)

#st.write(r)
classe = r["classe"]
prevision = r['prevision']

st.markdown(""" <br> <br> """, unsafe_allow_html=True)
st.write(f"Le client dont l'identifiant est **{data}** a obtenu le score de **{round(prevision*100,2):}%**.") #**{prevision:.1f}%**.")
st.write(f"**Il y a donc un risque de {prevision:.1f}% que le client ait des difficultés de paiement.**") #.1f
st.write(f"Le client est donc considéré par *'Prêt à dépenser'* comme **{classe[0]}** ")

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

# Calcul des valeurs Shap
explainer_shap = shap.TreeExplainer(model_LGBM)
#st.write(X_bis)
shap_values = explainer_shap.shap_values(X_bis)

st.header('Feature Importance')
plt.title("Feature importance bases on SHAP values")
shap.summary_plot(shap_values,X_bis)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot() #showPyplotGlobalUse = false
st.write('------------')

#data Id client
# Graphique force_plot
st.write("Le graphique suivant appelé `force-plot` permet de voir où se place la prédiction (f(x)) par rapport à la `base value`.")
st.write("Nous observons également quelles sont les variables qui augmentent la probabilité du client d'être \
        en défaut de paiement (en rouge) et celles qui la diminuent (en bleu), ainsi que l’amplitude de cet impact.")





st.write("Le graphique suivant appelé `force-plot` permet de voir où se place la prédiction (f(x)) par rapport à la `base value`.")
#generer la correspondance liste client
liste_id = list(X_bis.SK_ID_CURR)
idx_curr = liste_id.index(input_df)
#dict_id = zip(liste_id,idx)

print(idx_curr)
#idx_curr = dict_id["376270"]

#st_shap()
shap.decision_plot(explainer_shap.expected_value[1],
                   shap_values[1][idx_curr,:],
                   X_bis.iloc[idx_curr,:],
                   feature_names=X_bis.columns.to_list(),
                   feature_order='importance',
                   feature_display_range=slice(None, -15, -1),
                   link='logit')


st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
st.write('------------')


plt.title("Quel est l'Impact de chaque caractéristique sur la prédiction ?")

liste_variable()


pd.set_option('display.max_colwidth', None)

plt.title(" Comparaison du profil du client à celui des clients dont la probabilité de défaut de paiement est proche")

# Calcul des valeurs Shap
#explainer_shap = shap.TreeExplainer(model_LGBM)
#shap_values = explainer_shap.shap_values(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))

y_pred_lgbm = model_LGBM.predict(X_bis)  # Prédiction de la classe 0 ou 1
y_pred_lgbm_proba = model_LGBM.predict_proba(X_bis)  # Prédiction du % de risque

# Récupération du score du client
y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'], X_bis['SK_ID_CURR']], axis=1)

shap_values_df = pd.DataFrame(data=shap_values[1],
                              columns=X_bis.columns)

df_groupes = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'], shap_values_df], axis=1)
df_groupes['typologie_clients'] = pd.qcut(df_groupes.proba_classe_1,
                                          q=5,
                                          precision=1,
                                          labels=['20%_et_moins',
                                                  '21%_30%',
                                                  '31%_40%',
                                                  '41%_60%',
                                                  '61%_et_plus'])
# Titre H2
st.markdown("""
                <h2 style="color:#418b85;text-align:left;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
                Comparaison de “la trajectoire” prise par la prédiction du client à celles des groupes de Clients</h2>
                """,
            unsafe_allow_html=True)
st.write("")

# Moyenne des variables par classe
df_groupes_mean = df_groupes.groupby(['typologie_clients']).mean()
df_groupes_mean = df_groupes_mean.rename_axis('typologie_clients').reset_index()
df_groupes_mean["index"] = [1, 2, 3, 4, 5]
df_groupes_mean.set_index('index', inplace=True)

# récupération de l'index correspondant à l'identifiant du client
#idx = int(lecture_X_test_clean()[lecture_X_test_clean()['sk_id_curr'] == ID_client].index[0])

# dataframe avec shap values du client et des 5 groupes de clients
comparaison_client_groupe = pd.concat([df_groupes[df_groupes.index == idx_curr],
                                       df_groupes_mean],
                                      axis=0)
comparaison_client_groupe['typologie_clients'] = np.where(comparaison_client_groupe.index == idx_curr,
                                                          X_bis.iloc[idx_curr, 0],
                                                          comparaison_client_groupe['typologie_clients'])
# transformation en array
nmp = comparaison_client_groupe.drop(
    labels=['typologie_clients', "proba_classe_1"], axis=1).to_numpy()

fig = plt.figure(figsize=(8, 20))
shap.decision_plot(explainer_shap.expected_value[1],
                           nmp,
                           feature_names=comparaison_client_groupe.drop(
                               labels=['typologie_clients', "proba_classe_1"], axis=1).columns.to_list(),
                           feature_order='importance',
                           highlight=0,
                           legend_labels=['Client', '20%_et_moins', '21%_30%', '31%_40%', '41%_60%', '61%_et_plus'],
                           plot_color='inferno_r',
                           legend_location='center right',
                           feature_display_range=slice(None, -57, -1),
                           link='logit')
st.pyplot()






