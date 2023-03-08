import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import pickle
import lime
from lime import lime_tabular
import streamlit.components.v1 as components
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import dill
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
#import databases
import sqlalchemy
st.set_page_config(page_title = "Multipage App")
#DATABASE_URL = "sqlite:///./test.db"

"""host_server = os.environ.get('host_server', 'localhost')
db_server_port = urllib.parse.quote_plus(str(os.environ.get('db_server_port', '5432')))
database_name = os.environ.get('database_name', 'fastapi')
db_username = urllib.parse.quote_plus(str(os.environ.get('db_username', 'postgres')))
db_password = urllib.parse.quote_plus(str(os.environ.get('db_password', 'secret')))
ssl_mode = urllib.parse.quote_plus(str(os.environ.get('ssl_mode','prefer')))
DATABASE_URL = 'postgresql://{}:{}@{}:{}/{}?sslmode={}'.format(db_username, db_password, host_server, db_server_port, database_name, ssl_mode)


engine = sqlalchemy.create_engine(
    #DATABASE_URL, connect_args={"check_same_thread": False}
    DATABASE_URL, pool_size=3, max_overflow=0
)
metadata.create_all(engine)"""


st.set_option('deprecation.showPyplotGlobalUse', False)
#streamlit run /Users/alexandrejacqueline/Data_Science/P7/StreamlitApp/app.py


st.title("Application de Machine Learning pour ....")
st.subheader("Auteur : Alexandre Jacqueline")


def main():

    graph_perf = st.sidebar.multiselect(
    "Choisir un graphique",
    ("Confusion matrix","Roc Curve","Precision_Recall"))

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv("testing_data.csv")
        return data


    # Afficher la table de données
    df = load_data()
    data_load_state = st.text('Loading data...')
    df_sample = df.sample(100)
    if st.sidebar.checkbox("Afficher les données brutes", True):
        st.subheader("Jeu de donées: Echantillon de 100 observations")
        st.write(df_sample)
        option = st.selectbox("Wich client ?",df_sample['SK_ID_CURR'])
    #'You selected: ', option

    data_load_state.text('Loading data...done!')

    x_test = df.drop(columns=['TARGET'])
    y_test = df.TARGET

    class_names = ["Client with payment difficulties","Client without payment difficulties"]

    #analyse de la perf du modèle
    def plot_perf(graph):
        if "Confusion matrix" in graph:
            st.subheader('Matrice de confusion')
            plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
            st.pyplot()
        if "Roc Curve" in graph:
            st.subheader('Courbe ROC')
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        if "Precision_Recall" in graph:
            st.subheader('Courbe Recall et Precision')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()   # Precision_Recall



    #load modele
    model = pickle.load(open('lightgbm_Done.pkl','rb'))
    y_pred = model.predict(x_test)
    #Metrique
    accuracy = accuracy_score(y_test, y_pred).round(2)

    #Afficher les métriques dans l'app

    st.write("Accuracy", accuracy)

    #Afficher les graphiques de performances
    plot_perf(graph_perf)


if __name__ == '__main__':
    main()








