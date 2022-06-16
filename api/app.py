# from crypt import methods
from flask import Flask, render_template, request
from flask import jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///p7ocrDB.db'

# #On crée la database
# db = SQLAlchemy(app)

@app.route('/', methods=['get', 'post'])
def index():
    if request.method == 'POST':
        loan_id = request.form['loan_id']
        print("Numéro de prêt :", loan_id)
        df = pd.read_csv('static/test.csv')
        df.drop(columns=['Unnamed: 0'], inplace=True)
        data = df[df['SK_ID_CURR'] == int(loan_id)]

        # plt.figure(figsize = (6, 6))
        # plt.hist(x = df['CODE_GENDER'].values, bins = 50, align = 'left', color = '#4d7cab',
        #  edgecolor = 'black', linewidth = 1.1)
        # plt.axvline(x=0.5, color='r', label='axvline - full height')
        # plt.title('Distribution de notre colonne "CODE_GENDER"')
        # plt.xlabel('CODE_GENDER')
        # plt.ylabel('Count')
        # plt.show()
        return render_template("submitted.html", loan_id=loan_id, column_names=data.columns.values, row_data=list(data.values.tolist()),
                                link_column="SK_ID_CURR", zip=zip)
        
    else:
        return render_template("submitted.html")


# @app.route("/layout")
# def layout():
#     return render_template('layout.html', title='To Be Determined')

if __name__=='__main__':
    app.run(port=3000, debug=True)