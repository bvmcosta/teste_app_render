import os
import pickle 
import pandas as pd
from rossmann.Rossmann import Rossmann
from flask import Flask, request, Response
import xgboost as xgb

#Load XGBoost model - carrega o modelo em memória toda vez que a api for inicializada
model = pickle.load(open('model/model_rossmann2.pkl', 'rb'))

#Initialize 
app = Flask(__name__)

@app.route('/rossmann/predict', methods = ['POST'])
def rossmann_predict():

    test_json = request.get_json()

    if test_json: #there is data

        if isinstance(test_json, dict):
        
            test_raw = pd.DataFrame(test_json, index = [0]) #Funciona quando há apenas 1 linha
            
        else:
            
            test_raw = pd.DataFrame(test_json, columns = test_json[0].keys()) #Múltiplas linha

        #Instanciando a classe Rossmann - instanciar (criar uma cópia da classe, um exemplo)
        #Pacote - pasta onde está o script rosssmann.py
        pipeline = Rossmann()

        df1 = pipeline.data_cleaning(test_raw)
        df2 = pipeline.features_engineering(df1)
        df3 = pipeline.features_encoding_transformation(df2)
        df_response = pipeline.get_prediction(model, test_raw, df3)

        return df_response
        
    else:

        return Response('{}', status = 200, mimetype = 'application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host = '0.0.0.0', port = port,  debug = True) #Quando encontrar a função main, ele roda o método run no localhost (nosso computador)