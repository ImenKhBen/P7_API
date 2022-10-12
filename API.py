from flask import Flask, jsonify, request, jsonify, render_template
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from imblearn.under_sampling  import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier

# Create an instance of the Flask class that is the WSGI application.
# The first argument is the name of the application module or package, typically __name__ when using a single module.
app = Flask(__name__)

# import des modèles
X_test_init = joblib.load('X_test_init.joblib')
X_target = joblib.load('X_Target.joblib')

# On crée la liste des ID clients qui nous servira dans l'API (echantillon de 50 clients)
id_client = X_test_init["SK_ID_CURR"][:50].values
id_client = pd.DataFrame(id_client)

# routes
# Chargement des données pour la selection de l'ID client
@app.route("/load_data", methods=["GET"])
def load_data():  
    
    print("Dans fonction load_data")    
    return id_client.to_json(orient='values')

# Chargement d'informations générales
@app.route("/infos_gen", methods=["GET"])
def infos_gen():

    print("Dans fonction infos_gen")  
    lst_infos = [X_test_init.shape[0],
                 round(X_test_init["AMT_INCOME_TOTAL"].mean(), 2),
                 round(X_test_init["AMT_CREDIT"].mean(), 2)]

    return jsonify(lst_infos)

# Chargement des données pour le graphique dans la sidebar
@app.route("/disparite_target", methods=["GET"])
def disparite_target(): 

    print("Dans fonction disparite_target")  
    df_target = X_target["TARGET"].value_counts()

    return df_target.to_json(orient='values')

# Chargement d'informations générales sur le client
@app.route("/infos_client", methods=["GET"])
def infos_client():

    print("Dans fonction infos_client")  
    id = request.args.get("id_client")
    
    data_client = X_test_init[X_test_init["SK_ID_CURR"] == int(id)]
    
    print(data_client)
   
    response = json.loads(data_client.to_json(orient='index'))

    return response

# Calcul des ages de la population pour le graphique situant l'age du client
@app.route("/load_age_population", methods=["GET"])
def load_age_population():

    print("Dans fonction load_age_population")      
    df_age = round((X_test_init["DAYS_BIRTH"] / -365), 2)
    
    return df_age.to_json(orient='values')

# Segmentation des revenus de la population pour le graphique situant l'age du client
@app.route("/load_revenus_population", methods=["GET"])
def load_revenus_population():

    print("Dans fonction load_revenus_population")    
    # On supprime les outliers qui faussent le graphique de sortie
    df_revenus = X_test_init[X_test_init["AMT_INCOME_TOTAL"] < 700000]    
    df_revenus["tranches_revenus"] = pd.cut(df_revenus["AMT_INCOME_TOTAL"], bins=20)
    df_revenus = df_revenus[["AMT_INCOME_TOTAL", "tranches_revenus"]]
    df_revenus.sort_values(by="AMT_INCOME_TOTAL", inplace=True)

    print(df_revenus)
    
    df_revenus = df_revenus["AMT_INCOME_TOTAL"]

    return df_revenus.to_json(orient='values')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)