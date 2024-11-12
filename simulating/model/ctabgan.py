"""
Generative model training algorithm based on the CTABGANSynthesiser
"""

import pandas as pd
import time
from model.pipeline.data_preparation import DataPrep
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import warnings

warnings.filterwarnings("ignore")

class CTABGAN():

    def __init__(self,
                 raw_csv_path = "Real_Datasets/Adult.csv",
                 test_ratio = 0.20,
                 categorical_columns = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 general_columns = ["age"],
                 non_categorical_columns = [],
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": "income"},
                 epochs = 150,
                 batch_size = 500,
                 lr = 2e-4):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer(epochs=epochs,
                                              batch_size=batch_size,
                                              lr=lr)
        self.raw_df = pd.read_csv(raw_csv_path)
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type

    def fit(self):
        
        start_time = time.time()
        start_time2 = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
        print(f"Starting training: {start_time2}")
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.general_columns,self.non_categorical_columns,self.integer_columns,self.problem_type,self.test_ratio)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], mixed = self.data_prep.column_types["mixed"],
        general = self.data_prep.column_types["general"], non_categorical = self.data_prep.column_types["non_categorical"], type=self.problem_type)
        end_time = time.time()
        end_time2 = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
        print(f"Finished training: {end_time2}")
        print('Finished training in', time.strftime("%H:%M:%S", time.gmtime(end_time-start_time)))


    def generate_samples(self, n, var_column=None, var_class=None):

        if var_class != None:
            # Class index
            var_label_names = [var['column'] for var in self.data_prep.label_encoder_list]
            var_label_index = var_label_names.index(var_column)
            le = self.data_prep.label_encoder_list[var_label_index]['label_encoder']
            var_class_index = list(le.classes_).index(var_class)
            # Column index
            var_column_index = self.data_prep.df.columns.get_loc(var_column)
        else:
            var_column_index = None
            var_class_index = None
        
        sample = self.synthesizer.sample(n, var_column_index, var_class_index)
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df