import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
from collections import Counter
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s[%(name)s][%(levelname)s] %(message)s',
                    datefmt='[%Y-%m-%d][%H:%M:%S]')


logger = logging.getLogger(__name__)
path = '../../dataset/'
states_file = 'states.json'
with open(path+states_file, 'r',encoding='utf-8') as f:
    states = json.load(f)

def modify_state(state):
    """
    Modify wrong state for correct name
    
    Args:
        :state: State

    Returns:
        :state: State modified
    
    """
    
    if state in states.keys():
        return states[state]
    else:
        return state

class Preprocces():
    """Class to Preprocess data"""

    def __init__(self,data):
        self.data = data

    def encoder_variable(self,variable_ind,variable_dep,df,order = True):
        """
        Encoder variable for number of times it appears
        
        Args:
            :variable_ind: Variable independent
            :variable_dep: Variable dependent
            :order: True or False to sorted variables

        Returns:
            :label_order: Dict with variable and new value
        
        """
        
        if order:
            label= df.groupby([variable_ind])[variable_dep].mean().sort_values().index
            label_order = {x:y for y,x in enumerate(label,0)}
    
            return label_order
        else:
            label= df.groupby([variable_ind])[variable_dep].mean()
            label_encoder = label.to_dict()
            return label_encoder 
    

    def fill_categorical_na(self,columns,df,knn=False):
        """
        Impute categorical NaN with two methods
        
        Args:
            :columns: Columns to impute NaN
            :df: Dataframe
            :knn: True or False to use KNNImputer

        Returns:
            :df_result: Dataframe without NaN
        
        """
        df_result = df.copy()
        if not knn:

            df_result[columns] = df_result[columns].fillna('not indicated')
            return df_result
        else:
            self.state_ordered = self.encoder_variable('state','convert',df_result,True)
            self.industry_ordered = self.encoder_variable('industry','convert',df_result,True)
            self.business_ordered = self.encoder_variable('business_structure','convert',df_result,True)
            self.product_ordered = self.encoder_variable('product','convert',df_result,True)

            self.inv_state_orderd = {v:k for k,v in self.state_ordered.items()}
            self.inv_business_orderd = {v:k for k,v in self.business_ordered.items()}

            df_result['state_ordered']=df_result.state.map(self.state_ordered)
            df_result['industry_ordered']=df_result.industry.map(self.industry_ordered)
            df_result['business_ordered']=df_result.business_structure.map(self.business_ordered)
            df_result['product_ordered']=df_result['product'].map(self.product_ordered)

            imputer = KNNImputer(n_neighbors=2, weights="uniform")
            
            data_cat_imputed = imputer.fit_transform(df_result.drop(['account_uuid','product','state',
            'industry','subindustry','business_structure'],axis=1))

            columns = ['premium', 'carrier_id', 'convert','year_established', 'annual_revenue',
            'total_payroll', 'num_employees', 'state_ordered',
            'industry_ordered', 'business_ordered', 'product_ordered']

            for i in range(data_cat_imputed.shape[1]):
                df_result[columns[i]]=data_cat_imputed[:, i]

            return df_result


    def fill_numerical_na(self,columns,df):
        """
        Impute numerical NaN with IterativeImputer
        
        Args:
            :columns: Columns to impute NaN
            :df: Dataframe

        Returns:
            :df_result: Dataframe without NaN
        
        """
        
        imp = IterativeImputer(missing_values=np.nan,max_iter=15, random_state=0)

        imp.fit(df[columns])

        data_imputed = imp.transform(df[columns])
        for i in range(data_imputed.shape[1]):
            df[columns[i]]=data_imputed[:, i]
        df.year_established = round(df.year_established)
        df.num_employees = round(df.num_employees)
        
        return df
    
    def clean_dataframe(self):
        """Clean dataframe"""

        df = self.data
        logger.info('Shape of dataframe:'+ str(df.shape))
        logger.info(f'Remove outlier from column year_established')
        df = df.drop(df[(df.year_established>2021) | (df.year_established<1800)].index)
        logger.info('Shape of dataframe:'+ str(df.shape))
        logger.info(f'Transform column year_established to company years example: 2017 --> 4')
        df['year_established'] = 2021-df.year_established
        
        logger.info(f"Apply logarithm to annual_revenue total_payroll premium")

        df[['annual_revenue','total_payroll','premium']] = df[['annual_revenue','total_payroll','premium']].applymap(lambda x:np.log(x) if x>0 else x)

        return df


    def encode_and_bind(self,df, features_to_encode):
        """
        Encoder categorical feature using dummies
        
        Args:
            :df: Dataframe
            :features_to_encode: list with feature to encoder

        Returns:
            :df_result: dataframe with encoder feature and original 
        
        """

        dummies = pd.get_dummies(df[features_to_encode])
        df_result = pd.concat([dummies, df], axis=1)
        df_result = df_result.drop(features_to_encode, axis=1)
        return df_result


    











