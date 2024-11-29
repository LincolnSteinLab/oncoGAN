#!/genomeGAN/venvGAN/bin/python

"""
Generative model training algorithm based on the CTABGANSynthesiser
"""

import sys
sys.path.append('/genomeGAN/training/ctabgan/')

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
                 lr = 2e-4,
                 tqdm_disable = False):

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
        self.tqdm_disable = tqdm_disable

    def fit(self):
        
        if not self.tqdm_disable:
            start_time = time.time()
            start_time2 = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
            print(f"Starting training: {start_time2}")
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.general_columns,self.non_categorical_columns,self.integer_columns,self.problem_type,self.test_ratio)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], mixed = self.data_prep.column_types["mixed"],
        general = self.data_prep.column_types["general"], non_categorical = self.data_prep.column_types["non_categorical"], type=self.problem_type, tqdm_disable=self.tqdm_disable)
        if not self.tqdm_disable:
            end_time = time.time()
            end_time2 = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
            print(f"Finished training: {end_time2}")
            print('Finished training in', time.strftime("%H:%M:%S", time.gmtime(end_time-start_time)))


    def generate_samples(self, n):

        sample = self.synthesizer.sample(n) 
        if len(sample) < n:
            return(sample)
        else:
            sample_df = self.data_prep.inverse_prep(sample)
            return sample_df