from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# On charge notre modèle de prédiction
clf = pickle.load(open('static/xgboostclassifier.pkl','rb'))

@app.route('/', methods=['get', 'post'])
def index():
    if request.method == 'POST':
        # On choisit le prêt
        loan_id = request.form['loan_id']

        # On charge nos données
        df_predict = pd.read_csv('static/data_preprocessed_sample.csv')
        df_predict.drop(columns=['Unnamed: 0', 'index'], inplace=True)

        # On vérifie que le prêt est valide
        if loan_id.isdigit():
            loan_id = int(loan_id)

            if loan_id in df_predict['SK_ID_CURR'].unique():
                
                # On ne veut que le prêt sélectionné et on fait nos prédictions
                data_predict = df_predict[df_predict['SK_ID_CURR'] == loan_id]
                data_predict.set_index('SK_ID_CURR', inplace=True)
                predictions_proba = clf.predict_proba(data_predict.drop('TARGET', axis = 1))
                predictions = clf.predict(data_predict.drop('TARGET', axis=1))

                # On ajoute ces prédictions pour notre visuel
                data_predict['PREDICTIONS_PROBA'] = predictions_proba[:, 0]
                data_predict['PREDICTIONS'] = predictions

                # Fonction coût-métier : seuil à 0.57
                data_predict.loc[data_predict['PREDICTIONS_PROBA'] >= 0.57, 'PREDICTIONS'] = 0
                data_predict.loc[data_predict['PREDICTIONS_PROBA'] < 0.57, 'PREDICTIONS'] = 1
                data_predict.reset_index(inplace=True)

                # On ne prend que les informations essentielles
                data = data_predict[['SK_ID_CURR', 'PREDICTIONS_PROBA', 'PREDICTIONS']]

                proba = predictions_proba[:, 0]
                proba = round(proba[0], 2)

                if proba >= 0.57:
                    decision = 'Le prêt devrait être accordé.'
                else:
                    decision = 'Le prêt ne devrait pas être accordé.'

                return render_template("submitted.html", loan_id=loan_id, proba=proba, decision=decision,
                                        column_names=data.columns.values, row_data=list(data.values.tolist()),
                                        link_column="SK_ID_CURR", zip=zip)
            else:
                return render_template('submitted_fail.html')
        else:
            return render_template('submitted_fail.html')

    else:
        return render_template("submitted.html")

if __name__=='__main__':
    app.run(debug=True)
