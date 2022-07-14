import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import shap
import streamlit as st

# On supprime les avertissements nous indiquant que l'on change les valeurs de notre jeu de données d'origine
pd.options.mode.chained_assignment = None

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Prêt à dépenser : dashboard relation client")

st.number_input("ID du prêt", min_value=100002, value=100002, step=1, format="%d", key="loan_id")
loan_id = st.session_state.loan_id

@st.cache
def load_data():
    df_client = pd.read_csv('static/infos_clients_test.csv')
    df_client.drop(columns=['Unnamed: 0'], inplace=True)

    df_pret = pd.read_csv('static/infos_prets_test.csv')
    df_pret.drop(columns=['Unnamed: 0'], inplace=True)

    df_predict = pd.read_csv('static/data_preprocessed_sample.csv')
    df_predict.drop(columns=['Unnamed: 0', 'index'], inplace=True)
    return df_client, df_pret, df_predict

# On charge le jeu de données pour nos visuels et le jeu de données pour les prédictions 
df_client, df_pret, df_predict = load_data()

if loan_id in df_predict['SK_ID_CURR'].unique():

    # On s'intéresse à un prêt en particulier
    data_pret = df_pret[df_pret['ID_PRET'] == int(loan_id)]
    data_clt = df_client[df_client['ID_PRET'] == int(loan_id)]
    data_predict = df_predict[df_predict['SK_ID_CURR'] == int(loan_id)]

    # Et à ses voisins si nécessaire
    df_neighbors = df_predict.drop(columns='TARGET')
    clt = data_predict.drop(columns='TARGET').values
    distance = np.square(df_neighbors - clt).sum(axis=1)

    # On charge notre modèle de prédiction
    clf = pickle.load(open('static/xgboostclassifier.pkl','rb'))

    # Nos prédictions et leurs probabilités
    data_predict.set_index('SK_ID_CURR', inplace=True)

    predictions_proba = clf.predict_proba(data_predict.drop('TARGET', axis = 1))
    predictions = clf.predict(data_predict.drop('TARGET', axis = 1))

    # On ajoute ces prédictions pour nos visuels
    data_pret['PREDICTIONS_PROBA'] = predictions_proba[:, 0]
    data_pret['PREDICTIONS'] = predictions

    data_pret.loc[data_pret['PREDICTIONS_PROBA'] >= 0.57, 'PREDICTIONS'] = 0
    data_pret.loc[data_pret['PREDICTIONS_PROBA'] < 0.57, 'PREDICTIONS'] = 1

    image = Image.open('static/logo_pret_a_depenser.png')

    with st.sidebar:

        st.image(image)

        # On transpose pour avoir un meilleur visuel
        data_clt = data_clt.transpose()       

        if st.checkbox('Montrer information client ?'):
            st.subheader('Informations client :')
            st.dataframe(data_clt.astype(str))

        n_neighbors = st.slider("Nombre de voisins :", min_value=1, max_value=30, value=10)
        neighbors = (-distance).nlargest(n_neighbors)
        neighbors = neighbors.index.values.tolist()

        if st.checkbox('Numéros de prêts similaires :'):
            # Et à ses voisins
            st.write('Liste des prêts similaires :', neighbors)

    if st.checkbox('Montrer information prêt ?'):
        st.subheader('Informations prêt :')
        st.dataframe(data_pret.astype(str))

    st.subheader('Scoreboard du prêt sélectionné')

    # Premier graph
    col1, col2, col3, col4 = st.columns(4)

    with col2:

        fig1 = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = data_pret['PREDICTIONS_PROBA'].values[0],
            delta = {'reference': 0.57},
            gauge = {'axis': {'range': [None, 1]},
                    'steps' : [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.64], 'color': "gray"}],
                    'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.9, 'value': 0.57}},
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilité de remboursement du prêt"}))

        st.plotly_chart(fig1)

    data_pret.drop(columns=['PREDICTIONS_PROBA', 'PREDICTIONS'], inplace=True)

    # Deuxième graph

    # data_neighbors = df_neighbors[df_neighbors['SK_ID_CURR'].isin(neighbors)]

    st.subheader('Distribution des revenus et revenus du client')

    # Copie pour faire des changements au cas où on aurait besoin de l'initial

    # On choisit entre tous les clients et seulement quelques voisins

    if st.checkbox('Regarder la distribution des clients similaires :'):
        # Nos options

        options = ['AGE', 'REVENU_TOTAL', 'DETTE_TOTALE']
        option_bis = ['MONTANT_CREDIT', 'MONTANT_ANNUITE', 'SCORE_EXT_1', 'SCORE_EXT_2', 'SCORE_EXT_3']
        options.extend(option_bis)
        option = st.selectbox('Variable à choisir :', tuple(options))
    
        if option in option_bis:
            df_change_prets = df_pret.copy()
            df_change_prets = df_change_prets[df_change_prets['ID_PRET'].isin(neighbors)]

            fig2, ax = plt.subplots()
            ax.set_xlabel('Clients', fontsize=17)
            ax.set_ylabel(option, fontsize=17)
            client = (df_pret[df_pret['ID_PRET']== int(loan_id)][option]).values[0]
            ax.axhline(y=client, color='r', label='axhline - full height')
            ax = plt.boxplot(df_change_prets[option], showfliers=False)

            st.pyplot(fig2)
        
        else:
            df_change_clts = df_client.copy()
            df_change_clts = df_change_clts[df_change_clts['ID_PRET'].isin(neighbors)]

            fig2, ax = plt.subplots()
            ax.set_xlabel('Clients', fontsize=17)
            ax.set_ylabel(option, fontsize=17)
            client = (df_client[df_client['ID_PRET']== int(loan_id)][option]).values[0]
            ax.axhline(y=client, color='r', label='axhline - full height')
            ax = plt.boxplot(df_change_clts[option], showfliers=False)

            st.pyplot(fig2)

    if st.checkbox('Regarder la distribution de tous nos clients :'):

        options = ['AGE', 'REVENU_TOTAL', 'DETTE_TOTALE']
        option_bis = ['MONTANT_CREDIT', 'MONTANT_ANNUITE', 'SCORE_EXT_1', 'SCORE_EXT_2', 'SCORE_EXT_3', 'RETARD_PAIEMENT_MAX', 'RETARD_PAIEMENT_TOTAL']
        options.extend(option_bis)
        option = st.selectbox('Variable à choisir :', tuple(options))
    
        if option in option_bis:

            fig2, ax = plt.subplots()
            ax.set_xlabel('Clients', fontsize=17)
            ax.set_ylabel(option, fontsize=17)
            client = (df_pret[df_pret['ID_PRET']== int(loan_id)][option]).values[0]
            ax.axhline(y=client, color='r', label='axhline - full height')
            ax = plt.boxplot(df_pret[option], showfliers=False)

            st.pyplot(fig2)
        
        else:

            fig2, ax = plt.subplots()
            ax.set_xlabel('Clients', fontsize=17)
            ax.set_ylabel(option, fontsize=17)
            client = (df_client[df_client['ID_PRET']== int(loan_id)][option]).values[0]
            ax.axhline(y=client, color='r', label='axhline - full height')
            ax = plt.boxplot(df_client[option], showfliers=False)

            st.pyplot(fig2)


    # Troisième graph
    st.subheader('Les variables les plus importantes')

    # list_index_val = [x for x in range(df_predict.shape[0])]
    # df_predict['INDEX_VAL'] = list_index_val

    # index_value = df_predict.loc[df_predict['SK_ID_CURR'] == loan_id, 'INDEX_VAL'].values[0]
    # df_predict.drop(columns=['INDEX_VAL'], inplace=True)

    # On charge notre explainer
    explainer = pickle.load(open('static/shap_explainer.pkl','rb'))

    # values = data_predict.iloc[0, :].to_frame().transpose()

    shap_values = explainer(data_predict.drop(columns='TARGET'))
    shap_values.values = shap_values.values.reshape(-1)
    shap_values.base_values = shap_values.base_values[0]
    shap_values.data = shap_values.data.reshape(-1)

    fig3, ax = plt.subplots()

    ax.set_xlabel('\nImportance des variables dans la décision d\'octroi de prêt', fontsize=17)
    ax.set_ylabel('Variables', fontsize=17)

    l = st.slider("Nombre de variables à afficher", min_value=1, max_value=15, value=5)
    fig3 = shap.plots.waterfall(shap_values, max_display=l)
    st.pyplot(fig3)

else:
    st.write('Attention, le numéro de prêt est invalide.')