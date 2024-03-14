# Import libraries
from flask import Flask, render_template, request
from sqlalchemy import create_engine
from urllib.parse import quote
import pandas as pd
import pickle
import joblib

imp_scale = joblib.load('procesed.pkl')  # Imputation and Scaling pipeline
model = pickle.load(open('model.pkl', 'rb')) # KMeans clustering model
winsor = joblib.load('winsor')


def kmeans(data_new):
    clean1 = pd.DataFrame(imp_scale.transform(data_new), columns = imp_scale.get_feature_names_out())
    clean1[['numerical__Alcohol', 'numerical__Malic_Acid', 'numerical__Ash',
       'numerical__Ash_Alcanity', 'numerical__Magnesium',
       'numerical__Total_Phenols', 'numerical__Flavanoids',
       'numerical__Nonflavanoid_Phenols', 'numerical__Proanthocyanins',
       'numerical__Color_Intensity', 'numerical__Hue', 'numerical__OD280',
       'numerical__Proline']] = winsor.transform(clean1[['numerical__Alcohol', 'numerical__Malic_Acid', 'numerical__Ash',
       'numerical__Ash_Alcanity', 'numerical__Magnesium',
       'numerical__Total_Phenols', 'numerical__Flavanoids',
       'numerical__Nonflavanoid_Phenols', 'numerical__Proanthocyanins',
       'numerical__Color_Intensity', 'numerical__Hue', 'numerical__OD280',
       'numerical__Proline']])
   
    prediction = pd.DataFrame(model.predict(clean1), columns = ['Kmeans_clusters'])
   
    final_data = pd.concat([prediction, data_new], axis = 1)
    return(final_data)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        user = 'root'
        pw = 'Goldroger'
        db = 'hem'
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
     
        try:

            data = pd.read_csv(f)
        except:
                try:
                    data = pd.read_excel(f)
                except:      
                    data = pd.DataFrame(f)
                    data.to_sql('wine', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
                  
       
        html_table = data.to_html(classes = 'table table-striped')

        return render_template("data.html", Y = f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #888a9e;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}")

if __name__=='__main__':
    app.run(debug = True)
