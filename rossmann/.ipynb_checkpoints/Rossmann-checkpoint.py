#Bibliotecas
import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime


class Rossmann( object ):

    def __init__( self ): #Primeira função que a classe roda quando é instanciada - construtor.
        
        #self - salvará dentro da própria classe
        #self - variáveis dentro da classe rossmann que não podem ser acessadas sem utilizar a função - utilizando conceito de programação orientada a objeto
        #self - se uma classe externa quer acessar parâmetros internos tem que ser via método
        self.competition_distance_scaler    = pickle.load(open('../parameter/competition_distance_scaler.pkl', 'rb'))
        
        self.competition_time_months_scaler = pickle.load(open('../parameter/competition_time_months_scaler.pkl', 'rb'))

        self.promo_time_week_scaler         = pickle.load(open('../parameter/promo_time_week_scaler.pkl', 'rb'))

        self.year_scaler                    = pickle.load(open('../parameter/year_scaler.pkl', 'rb'))

        self.store_type_scaler              = pickle.load(open('../parameter/store_type_scaler.pkl', 'rb'))
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    def data_cleaning( self, df1 ):

        columns = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment', 
                   'CompetitionDistance', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek', 'Promo2SinceYear', 
                   'PromoInterval']

        snakecase = lambda column: inflection.underscore(column)

        columns = list(map(snakecase, columns))

        df1.columns = columns

        df1['date'] = pd.to_datetime(df1['date'])
        
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 100000 if math.isnan(x) else x)

        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else 
                                                        x['competition_open_since_month'], axis=1)
        
        df1['competition_open_since_year']  = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else 
                                                        x['competition_open_since_year'], axis=1)

        df1['promo2_since_week']  = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis=1)
        
        df1['promo2_since_year']  = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis=1)

        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec' }
        df1.loc[df1['promo_interval'].isnull() == True, 'promo_interval'] = 0
        df1['month_map'] = df1['date'].dt.month.map(month_map)
        df1['is_promo']  = df1[['month_map', 'promo_interval']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in 
                                                                      x['promo_interval'].split(',') else 0, axis=1)

        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

        return df1
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    def features_engineering(self, df1):
        
        #year
        df1['year'] = (df1['date'].dt.year).astype(int)
        #month
        df1['month'] = (df1['date'].dt.month).astype(int)
        #day
        df1['day'] = (df1['date'].dt.day).astype(int)
        #week of year
        #df1['week_of_year'] = (df1['date'].dt.isocalendar().week().astype(int)
        #A função isocalendar().week() atribui valor 1 à semana do dia 31-12-2013 e 2014. Isso pode agrupar as semanas no começo de um ano e começo do ano
        #seguinte
            
        #year_week
        df1['year_week'] = df1['date'].dt.strftime('%Y-%W')
        df1['week_of_year'] = df1['year_week'].str.split('-').str[1].astype(int)
    
        #competition and promo since
        df1['competition_since'] = df1.apply(lambda x: datetime.datetime(year = x['competition_open_since_year'], 
                                                                       month = x['competition_open_since_month'],
                                                                       day =1), axis = 1)
    
        df1['competition_time_months'] = ((df1['date'] - df1['competition_since'])/30).apply(lambda x: x.days).astype(int)
        #--------------------------------------------------------------------------------------------------------------------
        df1['promo_since'] = df1['promo2_since_year'].astype(str) + '-' +  df1['promo2_since_week'].astype(str)
        df1['promo_since'] = df1['promo_since'].apply(lambda x: datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days = 7))
        df1['promo_time_week']  = ((df1['date'] - df1['promo_since'])/7).apply(lambda x: x.days).astype(int)
        #--------------------------------------------------------------------------------------------------------------------
        df1['assortment'] = df1['assortment'].apply(lambda x: 'basic' if x == 'a' else 'extra' if x == 'b' else 'extended')
        #--------------------------------------------------------------------------------------------------------------------
        df1['state_holiday'] = df1['state_holiday'].apply(lambda x: 'Public_holiday' if x == 'a' else 'Easter_holiday' if x == 'b' else 'Christmas' if x == 'c' 
                                                          else 'Regular_day')

        df1 = df1[(df1['open'] != 0)].copy()
        cols_drop = ['open', 'promo_interval', 'month_map'] #Remover a coluna 'open' pq todos os valores são 1
                                                            #Podem ser removidas as colunas 'promo_interval' e 'month_map' pq foram usadas para derivar a 
                                                            #coluna 'is_promo'
        df1 = df1.drop(cols_drop, axis=1)

        return df1
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    def features_encoding_transformation(self, df1):
    
        df1 = df1.drop(['promo_since', 'competition_since'], axis = 1)

        #CYCLIC VARIABLES
        ## 'day'
        df1['day_sin'] = df1['day'].apply(lambda x: np.sin(x * (2 * np.pi / 30 )))
        df1['day_cos'] = df1['day'].apply(lambda x: np.cos(x * (2 * np.pi / 30 )))
    
        ## 'day_of_week
        df1['day_of_week_sin'] = df1['day_of_week'].apply(lambda x: np.sin(x * (2 * np.pi / 7 )))
        df1['day_of_week_cos'] = df1['day_of_week'].apply(lambda x: np.cos(x * (2 * np.pi / 7 )))
    
        ## 'month'
        df1['month_sin'] = df1['month'].apply(lambda x: np.sin(x * (2 * np.pi / 12 )))
        df1['month_cos'] = df1['month'].apply(lambda x: np.cos(x * (2 * np.pi / 12 )))
    
        ## 'week_of_year'
        df1['week_of_year_sin'] = df1['week_of_year'].apply(lambda x: np.sin(x * (2 * np.pi / 52 )))
        df1['week_of_year_cos'] = df1['week_of_year'].apply(lambda x: np.cos(x * (2 * np.pi / 52 )))
    
        cols_drop = ['day', 'day_of_week', 'month', 'week_of_year']
        df1 = df1.drop(cols_drop, axis = 1)
    
        #CATEGORICAL VARIABLES
    
        ## 'state_holiday'
        df1 = pd.get_dummies(df1, prefix = ['state_holiday'], columns = ['state_holiday'], dtype = int)
    
        ## 'store_type'
        df1['store_type'] = self.store_type_scaler.fit_transform(df1['store_type'])
    
        ## 'assortment'
        assortment = {'basic': 1, 'extended': 2, 'extra': 3}
        df1['assortment'] = df1['assortment'].map(assortment)
    
        #NUMERICAL VARIABLES    
        ## 'competition_distance'
        df1['competition_distance'] = self.competition_distance_scaler.fit_transform(df1[['competition_distance']].values)
           
        ## 'competition_time_months'
        df1['competition_time_months'] = self.competition_time_months_scaler.fit_transform(df1[['competition_time_months']].values)
        
        ## 'promo_time_week'
        df1['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df1[['promo_time_week']].values)
    
        ## 'year'
        df1['year'] = self.year_scaler.fit_transform(df1[['year']].values)

        columns_selected = ['competition_distance', 'competition_time_months', 'promo_time_week', 'store', 'store_type', 'assortment', 'promo',  'promo2', 
                            'day_sin', 'day_cos', 'day_of_week_sin', 'day_of_week_cos', 'week_of_year_cos', 'week_of_year_sin', 'month_sin', 'month_cos']

        return df1[columns_selected]
    #---------------------------------------------------------------------------------------------------------------------------------------------------
    def get_prediction( self, model, original, test ):

        prediction = model.predict(test)

        original['prediction'] = np.expm1(prediction)

        return original.to_json(orient = 'records', date_format = 'iso')