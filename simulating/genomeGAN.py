#!/usr/local/bin/python3

import sys
sys.path.append('/genomeGAN/')

import os
import re
import click
import pickle
import torch
import random
import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
from pyfaidx import Fasta


def tumor_models(tumor, device) -> list:

    """
    Get the specific models for the selected tumor type
    """

    # Counts model
    if tumor == "Breast-AdenoCa":
        countModel:dict = {}
        countModel['x1'] = torch.load(f"/genomeGAN/trained_models/counts/Breast-AdenoCa1_counts.pkl", map_location=device)
        countModel['x2'] = torch.load(f"/genomeGAN/trained_models/counts/Breast-AdenoCa2_counts.pkl", map_location=device)
    elif tumor == "CNS-PiloAstro":
        countModel:dict = {}
        countModel['x1'] = torch.load(f"/genomeGAN/trained_models/counts/CNS-PiloAstro1_counts.pkl", map_location=device)
        countModel['x2'] = torch.load(f"/genomeGAN/trained_models/counts/CNS-PiloAstro2_counts.pkl", map_location=device)
    elif tumor == "Eso-AdenoCa":
        countModel:dict = {}
        countModel['x1'] = torch.load(f"/genomeGAN/trained_models/counts/Eso-AdenoCa1_counts.pkl", map_location=device)
        countModel['x2'] = torch.load(f"/genomeGAN/trained_models/counts/Eso-AdenoCa2_counts.pkl", map_location=device)
        countModel['x3'] = torch.load(f"/genomeGAN/trained_models/counts/Eso-AdenoCa3_counts.pkl", map_location=device)
    elif tumor == "Liver-HCC":
        countModel:dict = {}
        countModel['x1'] = torch.load(f"/genomeGAN/trained_models/counts/Liver-HCC1_counts.pkl", map_location=device)
        countModel['x2'] = torch.load(f"/genomeGAN/trained_models/counts/Liver-HCC2_counts.pkl", map_location=device)
    elif tumor == "Lymph-CLL":
        countModel:dict = {}
        countModel['MUT']:dict = {}
        countModel['MUT']['x1'] = torch.load(f"/genomeGAN/trained_models/counts/Lymph-MCLL1_counts.pkl", map_location=device)
        countModel['MUT']['x2'] = torch.load(f"/genomeGAN/trained_models/counts/Lymph-MCLL2_counts.pkl", map_location=device)
        countModel['UNMUT']:dict = {}
        countModel['UNMUT']['x1'] = torch.load(f"/genomeGAN/trained_models/counts/Lymph-UCLL1_counts.pkl", map_location=device)
        countModel['UNMUT']['x2'] = torch.load(f"/genomeGAN/trained_models/counts/Lymph-UCLL2_counts.pkl", map_location=device)
    else:
        countModel = torch.load(f"/genomeGAN/trained_models/counts/{tumor}_counts.pkl", map_location=device)
    
    # Mutations model
    if tumor == "Lymph-CLL":
        mutModel:dict = {}
        mutModel['MUT'] = torch.load(f"/genomeGAN/trained_models/mutations/Lymph-CLL_mutations.pkl", map_location=device)
        mutModel['UNMUT'] = torch.load(f"/genomeGAN/trained_models/mutations/Lymph-CLL_mutations.pkl", map_location=device)
    else:
        mutModel = torch.load(f"/genomeGAN/trained_models/mutations/{tumor}_mutations.pkl", map_location=device)

    # Drivers model and files
    if tumor == "Lymph-CLL":
        driversModel:dict = {}
        driversModel['MUT']:dict = {}
        driversModel['MUT']['model'] = torch.load(f"/genomeGAN/trained_models/drivers/Lymph-MCLL_drivers.pkl", map_location=device)
        driversModel['MUT']['mutations'] = pd.read_csv(f"/genomeGAN/trained_models/drivers/Lymph-MCLL_driver_mutations.csv")
        driversModel['UNMUT']:dict = {}
        driversModel['UNMUT']['model']:dict = {}
        driversModel['UNMUT']['model']['x1'] = torch.load(f"/genomeGAN/trained_models/drivers/Lymph-UCLL1_drivers.pkl", map_location=device)
        driversModel['UNMUT']['model']['x2'] = torch.load(f"/genomeGAN/trained_models/drivers/Lymph-UCLL2_drivers.pkl", map_location=device)
        driversModel['UNMUT']['model']['x3'] = torch.load(f"/genomeGAN/trained_models/drivers/Lymph-UCLL3_drivers.pkl", map_location=device)
        driversModel['UNMUT']['mutations'] = pd.read_csv(f"/genomeGAN/trained_models/drivers/Lymph-UCLL_driver_mutations.csv")
    else:
        driversModel:dict = {}
        driversModel['model'] = torch.load(f"/genomeGAN/trained_models/drivers/{tumor}_drivers.pkl", map_location=device)
        driversModel['mutations'] = pd.read_csv(f"/genomeGAN/trained_models/drivers/{tumor}_driver_mutations.csv")

    # Positions model
    if tumor == "Lymph-CLL":
        posModel:dict = {}
        with open(f"/genomeGAN/trained_models/positions/Lymph-MCLL_positions.pkl", 'rb') as f:
           posModel['MUT'] = pickle.load(f)
        with open(f"/genomeGAN/trained_models/positions/Lymph-UCLL_positions.pkl", 'rb') as f:
           posModel['UNMUT'] = pickle.load(f)
    else:
        with open(f"/genomeGAN/trained_models/positions/{tumor}_positions.pkl", 'rb') as f:
           posModel = pickle.load(f)

    # Counts corrections
    countsCorr:pd.DataFrame = pd.read_csv(f"/genomeGAN/trained_models/counts/counts_correction_rates.csv")

    # Counts exclusions
    countsEx:pd.DataFrame = pd.read_csv(f"/genomeGAN/trained_models/counts/counts_exclusions.csv")

    return(countModel, mutModel, posModel, driversModel, countsCorr, countsEx)

def out_path(outDir, prefix, tumor, n) -> click.Path:

    """
    Get the absolute path and name for the outputs
    """

    if prefix is not None:
        output:click.Path = f"{outDir}/{prefix}_sim{n}.vcf"
    else:
        output:click.Path = f"{outDir}/{tumor}_sim{n}.vcf"
    
    return(output)

def preprocess_counts(counts, nCases, tumor, corrections, exclusions) -> pd.DataFrame:
    
    """
    Function to preprocess the counts
    """

    # Select corrections and exclusions tumor rows
    corrections = corrections.loc[corrections["tumor"]==tumor]
    corrections = corrections.drop('tumor', axis=1).reset_index(drop=True)
    exclusions = exclusions.loc[exclusions["tumor"]==tumor]
    exclusions = exclusions.drop('tumor', axis=1).reset_index(drop=True)

    # Assign a donor index
    counts['donor'] = [i for i in range(nCases*5)]
    
    # Pivot longer counts
    counts = counts.melt(id_vars=['donor'],var_name="mutations", value_name="count")

    # Left aligned counts and countsCorr
    counts = counts.merge(corrections, on='mutations', how='left')

    # Calculate total number of mutations per simulateed donor
    counts = counts.groupby('donor', group_keys=False).apply(lambda x: (x.assign(total=x['count'].sum(),
                                                                                 count_perc=x['count']/x['count'].sum()*100)))

    # Filter the counts
    counts = counts.groupby('donor', group_keys=False).apply(lambda x: (x.assign(filter_count_perc=np.where((x['count_perc'] < x['clean']),
                                                                                                            0,
                                                                                                            x['count_perc']))))
    counts = counts.groupby('donor', group_keys=False).apply(lambda x: (x.assign(filter_count_perc=np.where((x['filter_count_perc'] > x['max']) | ((x['filter_count_perc'] < x['min']) & ((x['filter_count_perc'] != 0))),
                                                                                                            np.nan,
                                                                                                            x['filter_count_perc']))))

    # Assign removed mutations
    counts = counts.groupby('donor', group_keys=False).apply(lambda x: (x.assign(removed=100-x['filter_count_perc'].sum())))
    counts = counts.groupby('donor', group_keys=False).apply(lambda x: (x.assign(updated_count_perc=x['filter_count_perc']+(x['filter_count_perc']/x['filter_count_perc'].sum()*x['removed']))))
    counts = counts.groupby('donor', group_keys=False).apply(lambda x: (x.assign(count=(x['updated_count_perc']*x['total']/100).round())))

    # Return the table to the original format
    counts = counts.drop(columns=['clean', 'max', 'min', 'total', 'count_perc', 'filter_count_perc', 'removed', 'updated_count_perc'])
    counts = counts.pivot(index='donor', columns='mutations', values='count').reset_index(drop=True).rename_axis(None, axis=1)
    counts = counts.dropna(axis=0, how='any').astype(int).reset_index(drop=True)
    
    # Check that there is a column for each mutation and if not initialize it
    counts = counts.assign(**{col:0 for col in ["DNP", "TNP", "INS", "DEL"] if col not in counts.columns})
    
    if exclusions.shape[0] > 0:
        for _, row in exclusions.iterrows():
            signatureA:str = row['signatureA']
            signatureB:str = row['signatureB']
            
            # Get the index to remove
            index_rm:pd.Index = counts[(counts[signatureA] > 0) & (counts[signatureB] > 0)].index
            
            # Remove the columns
            counts = counts.drop(index_rm).reset_index(drop=True)
    else:
        pass

    return(counts)

def simulate_counts(tumor, countSynthesizer, nCases, corrections, exclusions) -> pd.DataFrame:

    """
    Function to generate the number of each type of mutation per case
    """

    if tumor == "Breast-AdenoCa":
        # Model 1
        x1_counts:pd.DataFrame = pd.DataFrame()
        while x1_counts.shape[0] < nCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['x1'].generate_samples(nCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, nCases, tumor, corrections, exclusions)
            x1_counts = pd.concat([x1_counts,tmp_counts], ignore_index=True)
        ## Specific model filters
        x1_counts['total'] = x1_counts.sum(axis=1)
        x1_counts['keep'] = x1_counts.apply(lambda x: 
                                            np.where(x['SBS5'] / x['total'] < 0.5,
                                                     np.random.choice([True, False], p=[0.3, 0.7]),
                                                     True),
                                            axis=1)
        x1_counts['keep2'] = x1_counts.apply(lambda x: 
                                            np.where((x['SBS18'] / x['total'] > 0.05) & (x['SBS18'] / x['total'] < 0.1),
                                                      np.random.choice([True, False], p=[0.2, 0.8]),
                                                      True),
                                            axis=1)
        x1_counts['keep'] = x1_counts.apply(lambda x: 
                                            np.where(x['keep'] & x['keep2'],
                                                      True,
                                                      False),
                                            axis=1)
        x1_counts = x1_counts.loc[x1_counts["keep"]==True]
        x1_counts = x1_counts.drop(['keep', 'total'], axis=1).reset_index(drop=True)

        # Model 2
        x2_counts:pd.DataFrame = pd.DataFrame()
        while x2_counts.shape[0] < nCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['x2'].generate_samples(nCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, nCases, tumor, corrections, exclusions)
            x2_counts = pd.concat([x2_counts,tmp_counts], ignore_index=True)
        ## Specific model filters
        x2_counts['total'] = x2_counts.sum(axis=1)
        x2_counts['keep'] = x2_counts.apply(lambda x: 
                                            np.where(x['SBS1'] / x['total'] < 0.1,
                                                     np.random.choice([True, False], p=[0.3, 0.7]),
                                                     True),
                                            axis=1)
        x2_counts['keep2'] = x2_counts.apply(lambda x: 
                                            np.where((x['SBS18'] / x['total'] > 0.05) & (x['SBS18'] / x['total'] < 0.1),
                                                      np.random.choice([True, False], p=[0.2, 0.8]),
                                                      True),
                                            axis=1)
        x2_counts['keep'] = x2_counts.apply(lambda x: 
                                            np.where(x['keep'] & x['keep2'],
                                                     True,
                                                     False),
                                            axis=1)
        x2_counts = x2_counts.loc[x2_counts["keep"]==True]
        x2_counts = x2_counts.drop(['keep', 'total'], axis=1).reset_index(drop=True)
        
        # Merge
        counts:pd.DataFrame = pd.concat([x1_counts, x2_counts], ignore_index=True)
        ## Specific general filters
        counts['modify'] = np.random.choice([True, False], size=counts.shape[0], p=[0.7, 0.3])
        counts['SBS2'] = np.where(counts['modify'], counts['SBS2'], 0)
        counts['SBS13'] = np.where(counts['modify'], counts['SBS13'], 0)
        counts = counts.drop(columns=['modify'])

        # Return counts
        counts = counts.sample(n=nCases).reset_index(drop=True)
        counts.fillna(0, inplace=True)
        return(counts)
    
    elif tumor == "CNS-PiloAstro":
        # Model 1
        x1_counts:pd.DataFrame = pd.DataFrame()
        while x1_counts.shape[0] < nCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['x1'].generate_samples(nCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, nCases, tumor, corrections, exclusions)
            x1_counts = pd.concat([x1_counts,tmp_counts], ignore_index=True)
        ## Specific model filters
        x1_counts['total'] = x1_counts.sum(axis=1)
        x1_counts['keep'] = x1_counts.apply(lambda x: 
                                                np.where(x['total'] > 500,
                                                        np.where(~((x['SBS8'] != 0) | (x['SBS23'] != 0)),
                                                                np.random.choice([True, False], p=[0.7, 0.3]),
                                                                True),
                                                        np.where((x['SBS8'] != 0) | (x['SBS19'] != 0) | (x['SBS23'] != 0),
                                                                    True,
                                                                    np.random.choice([True, False]))),
                            axis=1)
        x1_counts = x1_counts.loc[x1_counts["keep"]==True]
        x1_counts = x1_counts.drop(['keep', 'total'], axis=1).reset_index(drop=True)

        # Model 2
        x2_counts:pd.DataFrame = pd.DataFrame()
        while x2_counts.shape[0] < nCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['x2'].generate_samples(nCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, nCases, tumor, corrections, exclusions)
            x2_counts = pd.concat([x2_counts,tmp_counts], ignore_index=True)
        ## Specific model filters
        x2_counts['total'] = x2_counts.sum(axis=1)
        x2_counts['keep'] = x2_counts.apply(lambda x: 
                                                np.where(x['total'] > 500,
                                                        False,
                                                        np.where((x['SBS8'] != 0) | (x['SBS19'] != 0) | (x['SBS23'] != 0),
                                                                    False,
                                                                    np.random.choice([True, False]))),
                            axis=1)
        x2_counts = x2_counts.loc[x2_counts["keep"]==True]
        x2_counts = x2_counts.drop(['keep', 'total'], axis=1).reset_index(drop=True)
        
        # Merge
        counts:pd.DataFrame = pd.concat([x1_counts, x2_counts], ignore_index=True)

        # Return counts
        counts = counts.sample(n=nCases).reset_index(drop=True)
        counts.fillna(0, inplace=True)
        return(counts)
    
    elif tumor == "Eso-AdenoCa":
        # Model 1
        x1_counts:pd.DataFrame = pd.DataFrame()
        while x1_counts.shape[0] < nCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['x1'].generate_samples(nCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, nCases, tumor, corrections, exclusions)
            x1_counts = pd.concat([x1_counts,tmp_counts], ignore_index=True)
        ## Specific model filters
        x1_counts['total'] = x1_counts.sum(axis=1)
        x1_counts['keep'] = x1_counts.apply(lambda x: 
                                            np.where((x['SBS2'] == 0) & (x['SBS13'] == 0) & (x['SBS28'] == 0),
                                                     True,
                                                     False),
                                            axis=1)
        x1_counts = x1_counts.loc[x1_counts["keep"]==True]
        x1_counts = x1_counts.drop(['keep', 'total'], axis=1).reset_index(drop=True)

        # Model 2
        x2_counts:pd.DataFrame = pd.DataFrame()
        while x2_counts.shape[0] < nCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['x2'].generate_samples(nCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, nCases, tumor, corrections, exclusions)
            x2_counts = pd.concat([x2_counts,tmp_counts], ignore_index=True)
        ## Specific model filters
        x2_counts['total'] = x2_counts.sum(axis=1)
        x2_counts['keep'] = x2_counts.apply(lambda x: 
                                            np.where(x['SBS2'] != 0,
                                                     True,
                                                     False),
                                            axis=1)
        x2_counts = x2_counts.loc[x2_counts["keep"]==True]
        x2_counts['keep'] = x2_counts.apply(lambda x: 
                                            np.where(x['SBS13'] != 0,
                                                     np.random.choice([True, False]),
                                                     False),
                                            axis=1)
        x2_counts = x2_counts.loc[x2_counts["keep"]==True]
        x2_counts = x2_counts.drop(['keep', 'total'], axis=1).reset_index(drop=True)

        # Model 3
        x3_counts:pd.DataFrame = pd.DataFrame()
        while x3_counts.shape[0] < nCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['x3'].generate_samples(nCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, nCases, tumor, corrections, exclusions)
            x3_counts = pd.concat([x3_counts,tmp_counts], ignore_index=True)
        ## Specific model filters
        x3_counts['total'] = x3_counts.sum(axis=1)
        x3_counts['keep'] = x3_counts.apply(lambda x: 
                                            np.where(x['SBS28'] != 0,
                                                     True,
                                                     False),
                                            axis=1)
        x3_counts = x3_counts.loc[x3_counts["keep"]==True]
        x3_counts = x3_counts.drop(['keep', 'total'], axis=1).reset_index(drop=True)
        
        # Merge
        counts:pd.DataFrame = pd.concat([x1_counts, x2_counts, x3_counts], ignore_index=True)
        ## Specific general filters
        counts['total'] = counts.sum(axis=1)
        counts['SBS17a'] = counts.apply(lambda x:
                                        np.where((x['total'] < 30000) & (x['SBS17a']/x['total'] < 0.1),
                                                 np.random.choice([x['SBS17a'], 0], p=[0.7, 0.3]),
                                                 x['SBS17a']),
                                        axis=1)
        counts['SBS17b'] = counts.apply(lambda x:
                                        np.where((x['total'] < 30000) & (x['SBS17b']/x['total'] < 0.2),
                                                 np.random.choice([x['SBS17b'], 0], p=[0.7, 0.3]),
                                                 x['SBS17b']),
                                        axis=1)
        counts['SBS28'] = counts.apply(lambda x:
                                       np.where(x['SBS28']/x['total'] > 0.25,
                                                x['SBS28']/4,
                                                np.where(x['SBS28']/x['total'] > 0.15,
                                                         x['SBS28']/2.5,
                                                         x['SBS28'])),
                                        axis=1)
        counts['keep'] = counts.apply(lambda x:
                                      np.where((x['SBS17a']/x['total'] > 0.15) & (x['SBS13'] != 0), 
                                               False,
                                               np.where((x['SBS17b']/x['total'] > 0.3) & (x['SBS13'] != 0),
                                                        False,
                                                        np.where((x['SBS17a'] != 0) & (x['SBS17b'] == 0), 
                                                                 False,
                                                                 True))),
                                    axis=1)
        counts = counts.loc[counts["keep"]==True]
        counts = counts.drop(['keep', 'total'], axis=1).reset_index(drop=True)

        # Return counts
        counts = counts.sample(n=nCases).reset_index(drop=True)
        counts.fillna(0, inplace=True)
        return(counts)
    
    elif tumor == "Kidney-RCC":
        counts:pd.DataFrame = pd.DataFrame()
        while counts.shape[0] <= nCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer.generate_samples(nCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, nCases, tumor, corrections, exclusions)
            counts = pd.concat([counts,tmp_counts], ignore_index=True)
        counts = counts.sample(n=nCases).reset_index(drop=True)

        # Return counts
        counts = counts.sample(n=nCases).reset_index(drop=True)
        counts.fillna(0, inplace=True)
        return(counts)

    elif tumor == "Liver-HCC":
        # Model 1
        x1_counts:pd.DataFrame = pd.DataFrame()
        while x1_counts.shape[0] < nCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['x1'].generate_samples(nCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, nCases, tumor, corrections, exclusions)
            x1_counts = pd.concat([x1_counts,tmp_counts], ignore_index=True)
        ## Specific model filters
        x1_counts = x1_counts.sample(frac = 0.90).reset_index(drop=True)

        # Model 2
        x2_counts:pd.DataFrame = pd.DataFrame()
        while x2_counts.shape[0] < nCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['x2'].generate_samples(nCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, nCases, tumor, corrections, exclusions)
            x2_counts = pd.concat([x2_counts,tmp_counts], ignore_index=True)
        ## Specific model filters
        x2_counts = x2_counts.loc[x2_counts["SBS8"]!=0]
        
        # Merge
        counts:pd.DataFrame = pd.concat([x1_counts, x2_counts], ignore_index=True)

        # Return counts
        counts = counts.sample(n=nCases).reset_index(drop=True)
        counts.fillna(0, inplace=True)
        return(counts)
    
    elif tumor == "Lymph-CLL":
        mCases:int = round(nCases*0.42)
        uCases:int = nCases - mCases

        # MCLL
        ## Model 1
        x1m_counts:pd.DataFrame = pd.DataFrame()
        while x1m_counts.shape[0] < mCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['MUT']['x1'].generate_samples(mCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, mCases, 'Lymph-MCLL', corrections, exclusions)
            x1m_counts = pd.concat([x1m_counts,tmp_counts], ignore_index=True)
        ## Specific model filters
        x1m_counts['total'] = x1m_counts.sum(axis=1)
        x1m_counts['keep'] = x1m_counts.apply(lambda x: 
                                                np.where((x['SBS8']/x['total']*100 >= 5) & (x['SBS8']/x['total']*100 <= 10),
                                                        False,
                                                        True),
                            axis=1)
        x2m_n_keep = x1m_counts.loc[x1m_counts["keep"]==False].shape[0]
        x1m_counts = x1m_counts.loc[x1m_counts["keep"]==True]
        x1m_counts = x1m_counts.drop(['keep', 'total'], axis=1).reset_index(drop=True)

        ## Model 2
        x2m_counts:pd.DataFrame = pd.DataFrame()
        while x2m_counts.shape[0] < mCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['MUT']['x2'].generate_samples(mCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, mCases, 'Lymph-MCLL', corrections, exclusions)
            x2m_counts = pd.concat([x2m_counts,tmp_counts], ignore_index=True)
        ### Specific model filters
        x2m_counts = x2m_counts.sample(n=x2m_n_keep)

        ## Merge
        m_counts:pd.DataFrame = pd.concat([x1m_counts, x2m_counts], ignore_index=True)
        m_counts = m_counts.sample(n=mCases).reset_index(drop=True)
        m_counts.fillna(0, inplace=True)

        # UCLL
        ## Model 1
        x1u_counts:pd.DataFrame = pd.DataFrame()
        while x1u_counts.shape[0] < mCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['UNMUT']['x1'].generate_samples(uCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, uCases, 'Lymph-UCLL', corrections, exclusions)
            x1u_counts = pd.concat([x1u_counts,tmp_counts], ignore_index=True)
        ### Specific model filters
        x1u_counts = x1u_counts.sample(frac=0.85)

        ## Model 2
        x2u_counts:pd.DataFrame = pd.DataFrame()
        while x2u_counts.shape[0] < uCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer['UNMUT']['x2'].generate_samples(uCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, uCases, 'Lymph-UCLL', corrections, exclusions)
            x2u_counts = pd.concat([x2u_counts,tmp_counts], ignore_index=True)
        ### Specific model filters
        x2u_counts['total'] = x2u_counts.sum(axis=1)
        x2u_counts['keep'] = x2u_counts.apply(lambda x: 
                                                np.where((x['SBS5']/x['total']*100 >= 75),
                                                        True,
                                                        False),
                            axis=1)
        x2u_counts = x2u_counts.loc[x2u_counts["keep"]==True]
        x2u_counts = x2u_counts.drop(['keep', 'total'], axis=1).reset_index(drop=True)

        ## Merge
        u_counts:pd.DataFrame = pd.concat([x1u_counts, x2u_counts], ignore_index=True)
        u_counts = u_counts.sample(n=uCases).reset_index(drop=True)
        u_counts.fillna(0, inplace=True)

        # Merge MCLL and UCLL
        counts:pd.DataFrame = pd.concat([m_counts, u_counts], ignore_index=True)
        counts = counts.sample(frac=1)
        counts.fillna(0, inplace=True)

        # Return counts
        return(counts)
        
def simulate_vaf_rank(tumor, nCases) -> list:

    """
    Function to simulate the VAF range for each donor
    """

    rank_file:pd.DataFrame = pd.read_csv("/genomeGAN/trained_models/vaf_rank_list.tsv", sep='\t') 
    rank_file = rank_file.loc[rank_file["study"]==tumor]
    donor_vafs:list = random.choices(rank_file.columns[1:], weights=rank_file.values[0][1:], k=nCases)

    return(donor_vafs)

def simulate_drivers(tumor, driversSynthesizer) -> pd.DataFrame:

    """
    Function to simulate the driver mutations for each donor
    """

    if tumor == "Breast-AdenoCa":
        drivers:pd.DataFrame = driversSynthesizer['model'].generate_samples(10)
        drivers = drivers.round(0).astype(int)
        drivers['ERBB4_Intron'] = drivers.apply(lambda x: 
                                                        np.where((x['ERBB4_Intron'] == 2) | (x['ERBB4_Intron'] == 5),
                                                                 np.random.choice([x['ERBB4_Intron'], x['ERBB4_Intron'] - 1], p=[0.7, 0.3]),
                                                                 x['ERBB4_Intron']),
                                                        axis=1)
        drivers['ERBB4_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['ERBB4_Intron'] != 0,
                                                                 np.random.choice([x['ERBB4_Intron'], 0], p=[0.6, 0.4]),
                                                                 x['ERBB4_Intron']),
                                                        axis=1)
        drivers['RUNX1_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['RUNX1_Intron'] != 0,
                                                                 np.random.choice([x['RUNX1_Intron'], 0], p=[0.4, 0.6]),
                                                                 x['RUNX1_Intron']),
                                                        axis=1)
        drivers['TP53_Coding'] = drivers.apply(lambda x: 
                                                        np.where(x['TP53_Coding'] != 0,
                                                                 np.random.choice([x['TP53_Coding'], 0], p=[0.5, 0.5]),
                                                                 x['TP53_Coding']),
                                                        axis=1)
        drivers['MAP3K1_Coding'] = drivers.apply(lambda x: 
                                                        np.where(x['MAP3K1_Coding'] != 0,
                                                                 np.random.choice([x['MAP3K1_Coding'], 0], p=[0.5, 0.5]),
                                                                 x['MAP3K1_Coding']),
                                                        axis=1)
        drivers['ARID1B_Coding'] = drivers.apply(lambda x: 
                                                        np.where(x['ARID1B_Coding'] != 0,
                                                                 np.random.choice([x['ARID1B_Coding'], 0], p=[0.6, 0.4]),
                                                                 x['ARID1B_Coding']),
                                                        axis=1)
        drivers['NOTCH2_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['NOTCH2_Intron'] != 0,
                                                                 np.random.choice([x['NOTCH2_Intron'], 0], p=[0.6, 0.4]),
                                                                 x['NOTCH2_Intron']),
                                                        axis=1)
        drivers['INPP4B_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['INPP4B_Intron'] != 1,
                                                                 np.random.choice([x['INPP4B_Intron'], 0], p=[0.3, 0.7]),
                                                                 x['INPP4B_Intron']),
                                                        axis=1)
        drivers['CBFB_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['CBFB_Intron'] != 1,
                                                                 np.random.choice([x['CBFB_Intron'], 0], p=[0.5, 0.5]),
                                                                 x['CBFB_Intron']),
                                                        axis=1)
        drivers['STAG2_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['STAG2_Intron'] != 1,
                                                                 np.random.choice([x['STAG2_Intron'], 0], p=[0.4, 0.6]),
                                                                 x['STAG2_Intron']),
                                                        axis=1)
        drivers['ANK3_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['ANK3_Intron'] != 3,
                                                                 np.random.choice([x['ANK3_Intron'], 0], p=[0.4, 0.6]),
                                                                 x['ANK3_Intron']),
                                                        axis=1)
    elif tumor == "Kideny-RCC":
        drivers:pd.DataFrame = driversSynthesizer['model'].generate_samples(10)
        drivers = drivers.round(0).astype(int)
        drivers['VHL_Coding'] = drivers.apply(lambda x: 
                                                        np.where(x['VHL_Coding'] != 0,
                                                                 np.random.choice([x['VHL_Coding'], 0], p=[0.8, 0.2]),
                                                                 x['VHL_Coding']),
                                                        axis=1)
        drivers['TSC2_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['TSC2_Intron'] != 0,
                                                                 np.random.choice([x['TSC2_Intron'], 0], p=[0.4, 0.6]),
                                                                 x['TSC2_Intron']),
                                                        axis=1)
        drivers['MET_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['MET_Intron'] != 0,
                                                                 np.random.choice([x['MET_Intron'], 0], p=[0.6, 0.4]),
                                                                 x['MET_Intron']),
                                                        axis=1)
    elif tumor == "Lymph-MCLL":        
        tmp_drivers:pd.DataFrame = driversSynthesizer['MUT']['model'].generate_samples(10)
        tmp_drivers = drivers.round(0).astype(int)
        tmp_drivers_CD36:pd.DataFrame = tmp_drivers[tmp_drivers['CD36_Intron'] != 1].reset_index(drop=True)
        tmp_drivers_CD36_1:pd.DataFrame = tmp_drivers[tmp_drivers['CD36_Intron'] == 1].sample(frac=0.4).reset_index(drop=True)
        drivers:pd.DataFrame = pd.concat([tmp_drivers_CD36,tmp_drivers_CD36_1], ignore_index=True)
    elif tumor == "Lymph-UCLL":
        x1_tmp_drivers:pd.DataFrame = driversSynthesizer['UNMUT']['model']['x1'].generate_samples(10)
        x2_tmp_drivers:pd.DataFrame = driversSynthesizer['UNMUT']['model']['x2'].generate_samples(10)
        x3_tmp_drivers:pd.DataFrame = driversSynthesizer['UNMUT']['model']['x3'].generate_samples(10)
        drivers:pd.DataFrame = pd.concat([x1_tmp_drivers, x2_tmp_drivers, x3_tmp_drivers], ignore_index=True)
    else:
        drivers:pd.DataFrame = driversSynthesizer['model'].generate_samples(10)

    drivers.fillna(0, inplace=True)
    drivers = drivers.round(0).astype(int)
    drivers = drivers.sample(frac=1).reset_index(drop=True)
    drivers = drivers.iloc[1]

    return(drivers)

def vaf_rank2float(vafs_rank_list) -> list:

    """
    Convert the VAF rank to a float value
    """

    final_vaf_list:list = []
    for vaf in vafs_rank_list:
        vaf:str = re.sub(r'[\[\)\]]', '', vaf)
        start:float = float(vaf.split(',')[0])
        end:float = float(vaf.split(',')[1])
        final_vaf_list.append(round(random.uniform(start, end), 2))

    return(final_vaf_list)

def filter_muts(file) -> pd.DataFrame: 

    """
    Remove incorrect mutations generated by the GAN
    """

    # Annotate the type of mutation
    conditions:list = [
        ((file['a.ctx1'] == file['r.ctx1']) & (file['a.ctx2'] != file['r.ctx2']) & (file['a.ctx3'] == file['r.ctx3']) & (file['a.ctx2'] != '.') & (file['signature'] != '.')),
        ((file['a.ctx1'] == file['r.ctx1']) & (file['a.ctx2'] != file['r.ctx2']) & (file['a.ctx3'] != file['r.ctx3']) & (file['a.ctx2'] != '.') & (file['a.ctx3'] != '.') & (file['signature'] == '.')),
        ((file['a.ctx1'] != file['r.ctx1']) & (file['a.ctx2'] != file['r.ctx2']) & (file['a.ctx3'] != file['r.ctx3']) & (file['a.ctx1'] != '.') & (file['a.ctx2'] != '.') & (file['a.ctx3'] != '.') & (file['signature'] == '.')),
        ((file['a.ctx1'] == '.') & (file['a.ctx2'] == '.') & (file['a.ctx3'] == '.') & (file['len'] > 0) & (file['signature'] == '.')),
        ((file['a.ctx1'] == '.') & (file['a.ctx2'] == '.') & (file['a.ctx3'] == '.') & (file['len'] < 0) & (file['signature'] == '.'))
    ]
    mutations:list = ['SNP', 'DNP', 'TNP', 'INS', 'DEL']
    file['mut'] = np.select(conditions, mutations, default='Error')
    file = file.loc[file["mut"] != "Error"]

    # Create the r.ctx and a.ctx columns to be used in the next step
    file["r.ctx"] = file["r.ctx1"]+file["r.ctx2"]+file["r.ctx3"]
    file["a.ctx"] = file["a.ctx1"]+file["a.ctx2"]+file["a.ctx3"]
    file.drop(["start", "vaf", "r.ctx1", "r.ctx2", "r.ctx3", "a.ctx1", "a.ctx2", "a.ctx3"], axis=1, inplace=True)

    return(file)

def manually_simulate_dnp_tnp(mutSynthesizer, nMut, mut_type, tumor) -> pd.DataFrame:
    
        """
        Function to manually simulate DNP and TNP mutations
        """
    
        nt_dict = {'A':['C', 'G', 'T'], 'C':['A', 'G', 'T'], 'G':['A', 'C', 'T'], 'T':['A', 'C', 'G']}
        if tumor == "Lymph-MCLL":
            muts = mutSynthesizer['MUT'].generate_samples(int(nMut)*2)
        elif tumor == "Lymph-UCLL":
            muts = mutSynthesizer['UNMUT'].generate_samples(int(nMut)*2)
        else:
            muts = mutSynthesizer.generate_samples(int(nMut)*2)
        muts = filter_muts(muts)
        muts = muts[muts['mut'] == 'SNP']
        muts = muts.take(indices=range(int(nMut)), axis=0)
        muts['mut'] = mut_type
        muts['signature'] = '.'
        if mut_type == 'DNP':
            muts['a.ctx'] = muts['a.ctx'].apply(lambda x:f"{x[0]}{x[1]}{random.choice(nt_dict[x[2]])}")
        else:
            muts['a.ctx'] = muts['a.ctx'].apply(lambda x:f"{random.choice(nt_dict[x[0]])}{x[1]}{random.choice(nt_dict[x[2]])}")
    
        return(muts)

def simulate_mutations(mutSynthesizer, muts, nMut, case_counts, tumor) -> pd.DataFrame:

    """
    Function to easily generate all the mutations for a case
    """

    # Initialize the variables
    generated_types:pd.Series = pd.Series()
    oriVSsim_types:pd.Series = pd.Series()
    first:bool = True
    rounds:int = 0

    # Enter in a loop until the number of mutations generated is the same as the number of mutations that have to be generated
    while (not all(oriVSsim_types==0)) or first:
        first = False
        ## Generate and filter the mutations
        if tumor == "Lymph-MCLL":
            tmp_muts:pd.DataFrame = mutSynthesizer['MUT'].generate_samples(int(round(nMut*30,0)))
        elif tumor == "Lymph-UCLL":
            tmp_muts:pd.DataFrame = mutSynthesizer['UNMUT'].generate_samples(int(round(nMut*30,0)))
        else:    
            tmp_muts:pd.DataFrame = mutSynthesizer.generate_samples(int(round(nMut*30,0)))
        tmp_muts = filter_muts(tmp_muts)
        muts = pd.concat([muts,tmp_muts])
        ## Check the number and type of the generated mutations
        tmp_generated_types = tmp_muts.apply(lambda x:x['mut'] if x['signature'] == '.' else x['signature'], axis=1).value_counts()
        tmp_generated_types, generated_types = tmp_generated_types.align(generated_types, fill_value=0)
        generated_types += tmp_generated_types
        case_counts, generated_types = case_counts.align(generated_types, fill_value=0)
        oriVSsim_types = case_counts - generated_types
        oriVSsim_types[oriVSsim_types < 0] = 0
        ## Break the loop if it is taking too long
        if rounds == 20:
            if oriVSsim_types['DNP'] > 0:
                dnp_muts = manually_simulate_dnp_tnp(mutSynthesizer, oriVSsim_types['DNP'], 'DNP', tumor)
                muts = pd.concat([muts,dnp_muts])
                oriVSsim_types['DNP'] = 0
            if oriVSsim_types['TNP'] > 0:
                tnp_muts = manually_simulate_dnp_tnp(mutSynthesizer, oriVSsim_types['TNP'], 'TNP', tumor)
                muts = pd.concat([muts,tnp_muts])
                oriVSsim_types['TNP'] = 0
            ## If there are still some mutations that are not simulated remove them from case_counts
            case_counts = case_counts - oriVSsim_types
            break
        rounds += 1

    muts.reset_index(drop=True, inplace=True)

    return(muts, case_counts)

def select_case_mutations(muts, case_counts) -> list:

    """
    Function to select the mutations from the whole mutations database for a case
    """
    
    case_muts:pd.DataFrame = pd.DataFrame()
    for mut,n in zip(case_counts.index, case_counts):
        if n == 0:
            continue
        else:
            if mut.startswith("SBS"):
                col = "signature"
            else:
                col = "mut"
            tmp:pd.DataFrame = muts[(muts[col]==mut)].take(indices=range(int(n)), axis=0)

            ## Update case mutations dataframe
            case_muts = pd.concat([case_muts,tmp])
            case_muts.reset_index(drop=True, inplace=True)
            ## Update muts dataframe
            muts.drop(labels=tmp.index, axis=0, inplace=True)
            muts.reset_index(drop=True, inplace=True)

    return(muts, case_muts)

def gender_selection(tumor) -> str:

    """
    Simulate the gender
    """

    if tumor in ["Breast-AdenoCa", "Breast-DCIS", "Breast-LobularCa", "Ovary-AdenoCA"]:
        gender:str = "F"
    elif tumor in ["Prost-AdenoCA"]:
        gender:str = "M"
    else:
        gender:str = "M" if random.random() < 0.5 else "F"
    
    return(gender)

def assign_chromosome(positions) -> pd.DataFrame:
    
    """
    Function to assign a chromosome to each position
    """
    
    # Positions are encoded in a continuous scale, so we need to decode them
    position_decode:np.array  = np.array([0,249250621,492449994,690472424,881626700,1062541960,1233657027,1392795690,1539159712,1680373143,1815907890,1950914406,2084766301,2199936179,2307285719,2409817111,2500171864,2581367074,2659444322,2718573305,2781598825,2829728720,2881033286,3036303846])
    chromosome_decode:np.array  = np.array(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y'])
    start_encoded:np.array  = np.digitize(positions["start"], position_decode, right=True)
    end_encoded:np.array  = np.digitize(positions["end"], position_decode, right=True)
    ## Assign new columns
    positions["start"] = positions["start"] - np.take(position_decode, start_encoded-1)
    positions["end"] = positions["end"] - np.take(position_decode, end_encoded-1)
    positions["start_chrom"] = np.take(chromosome_decode, start_encoded-1)
    positions["end_chrom"] = np.take(chromosome_decode, end_encoded-1)
    
    # Because the scale is continuous, some positions might be assigned to the next chromosome
    positions = positions[positions['start_chrom'] == positions['end_chrom']]
    positions.drop('end_chrom', axis=1, inplace=True)
    positions.rename(columns={'start_chrom':'chrom'}, inplace=True)

    return(positions)

def get_sequence(positions, fasta, posQueue=None) -> pd.DataFrame:
    
    """
    Function to get the DNA sequence of each position
    """
    
    for i, row in positions.iterrows():
        chrom:str = str(row['chrom'])
        start:int = int(row['start'])
        end:int = int(row['end'])
        sequence:str = fasta[chrom][start:end].seq
        positions.at[i, 'sequence'] = sequence

    if posQueue is not None:
        posQueue.put(positions)
    else:
        return(positions)

def get_sexual_chrom_usage(tumor, gender) -> dict:

    """
    Function to get the sexual chromosome usage
    """

    sexChrFile:pd.DataFrame = pd.read_csv("/genomeGAN/trained_models/xy_ranks.txt", sep='\t') 
    sexChrFile = sexChrFile.loc[sexChrFile["label"].isin([f'{tumor}_X{gender}', f'{tumor}_Y{gender}'])]
    sexChrFile['label'] = sexChrFile['label'].apply(lambda x:x[-2])

    sexChrDict:dict = {}
    for _,entry in sexChrFile.iterrows():
        rank:str = random.choice(entry['ranks'].split(';'))
        rankList:list = rank.strip('[').strip(']').split(',')
        rankList = [float(i) for i in rankList]
        rankValue:float = round(random.uniform(rankList[0], rankList[1]), 1)
        sexChrDict[entry['label']] = rankValue

    return(sexChrDict)

def get_coordinates(posSynthesizer, nMut) -> pd.DataFrame:

    """
    Function to generate the coordinates of the mutations
    """

    # Generate the windows
    step1:pd.DataFrame = posSynthesizer['step1'].sample(num_rows = nMut)
    step1 = step1['rank'].value_counts()

    # Generate the specific position ranges
    positions:pd.DataFrame = pd.DataFrame()
    for rank, n in zip(step1.index, step1):
        positions = pd.concat([positions, posSynthesizer[rank].sample(num_rows = n)])
    positions['start'] = positions['start']*100
    positions['end'] = positions['start']+100
    positions.reset_index(drop=True, inplace=True)

    # Annotate the chromosome
    positions = assign_chromosome(positions)

    return(positions)

def update_sexual_chrom_positions(positions, sexChrom, exp, posSynthesizer) -> pd.DataFrame:

    """
    Function to match the number of mutations in sex chromosome positions
    """

    # Split autosomal and sexual chromosomes
    normChr:pd.DataFrame = positions[positions['chrom'] != sexChrom]
    sexChr:pd.DataFrame = positions[positions['chrom'] == sexChrom]

    # Calculate the number of mutations previously simulated in the sexual chromosome
    obs:int = sexChr['chrom'].value_counts().get(sexChrom, 0)

    # Calculate how many mutations need to be udpated
    difference:int = obs-exp

    # Update mutations
    update:bool = True
    while difference != 0:
        if difference > 0: #More X/Y mutations than expected
            if update: #Remove excess mutations
                sexChr.drop(sexChr.sample(n=difference).index, inplace=True)
                sexChr.reset_index(drop=True, inplace=True)
                update = False
            ## Generate the new positions for autosomal chromosomes
            tmp_positions:pd.DataFrame = get_coordinates(posSynthesizer, difference*10)
            tmp_positions = tmp_positions[tmp_positions['chrom'] != sexChrom]
            ## Update positions database
            try:
                normChr = pd.concat([normChr, tmp_positions.sample(n=difference)])
                normChr.reset_index(drop=True, inplace=True)
                difference = 0
            except ValueError:
                normChr = pd.concat([normChr, tmp_positions])
                normChr.reset_index(drop=True, inplace=True)
                difference -= tmp_positions.shape[0]
        else: #Less X/Y mutations than expected
            if update: #Remove excess mutations
                normChr.drop(normChr.sample(n=abs(difference)).index, inplace=True)
                normChr.reset_index(drop=True, inplace=True)
                update = False
            ## Generate the new positions for sexual chromosomes
            tmp_positions:pd.DataFrame = get_coordinates(posSynthesizer, abs(difference)*100) #NOTE - This works, but for chromY could be faster
            tmp_positions = tmp_positions[tmp_positions['chrom'] == sexChrom]
            ## Update positions database
            try:
                sexChr = pd.concat([sexChr, tmp_positions.sample(n=abs(difference))])
                sexChr.reset_index(drop=True, inplace=True)
                difference = 0
            except ValueError:
                sexChr = pd.concat([sexChr, tmp_positions])
                sexChr.reset_index(drop=True, inplace=True)
                difference += tmp_positions.shape[0]
    positions = pd.concat([normChr, sexChr])
    positions.reset_index(drop=True, inplace=True)

    return(positions)

def generate_positions(posSynthesizer, nMut, fasta, cpus, sexChrDict) -> pd.DataFrame: 

    """
    Function to generate the positions of the mutations
    """

    # Get the coordinates of the future positions
    positions:pd.DataFrame = get_coordinates(posSynthesizer, nMut)

    # Update sexual chromosome positions
    for key in sexChrDict.keys():
        exp:int = int(round(nMut * sexChrDict[key] / 100, 0))
        positions = update_sexual_chrom_positions(positions, key, exp, posSynthesizer)
    
    # Identify the corresponding DNA sequence
    if cpus > 1:
        jobs:list = []
        posQueue = multiprocessing.Queue()

        # Split the file according to the number of available cpus
        posList:list = np.array_split(positions, cpus)

        # Create the jobs
        for pos in posList:
            p = multiprocessing.Process(target=get_sequence, args=(pos, fasta, posQueue))
            jobs.append(p)

        # Start the jobs
        for j in jobs:
            j.start()
        # Get the results from the queue
        posQueue_result:list = []
        while True:
            posRes = posQueue.get()
            posQueue_result.append(posRes)
            if len(posQueue_result) == cpus:
                break
        # Wait for the jobs to finish
        for j in jobs:
            j.join()
        # Concatenate the results
        positions = pd.concat(posQueue_result).reset_index(drop=True)
    else:
        positions = get_sequence(positions, fasta)

    # Shuffle the positions
    positions = positions.sample(frac=1).reset_index(drop=True)

    return(positions)

def assign_position(tumor, case_counts, case_muts, posModel, nMut, fasta, gender, cpus) -> pd.DataFrame:

    """
    Function to assign a position to each mutation
    """

    # posModel selection
    if tumor == "Lymph-CLL":
        if case_counts['SBS9'] > 0: #AID signature
            posModel = posModel['MUT']
        else:
            posModel = posModel['UNMUT']
    else:
        pass

    # Get sexual chromosome ranks
    sexChrDict:dict = get_sexual_chrom_usage(tumor, gender)

    # Initialize new columns with NaN values
    case_muts['chrom'] = pd.Series([np.nan]*len(case_muts))
    case_muts['pos'] = pd.Series([np.nan]*len(case_muts))

    # Assign positions
    while nMut > 0:
        ## Generate the positions
        positions:pd.DataFrame = generate_positions(posModel, nMut, fasta, cpus, sexChrDict)
        ## Find the correct one for the specific context
        for idx, mut in case_muts.loc[case_muts['chrom'].isna()].iterrows():
            r_ctx:str = mut['r.ctx']
            for pos_idx, pos in positions.iterrows():
                seq:str = pos['sequence']
                indexes:list = [m.start() for m in re.finditer(r_ctx, seq)]
                if indexes:
                    case_muts.at[idx, 'chrom'] = pos['chrom']
                    case_muts.at[idx, 'pos'] = pos['start']+random.choice(indexes)+2
                    positions.drop(pos_idx, axis=0, inplace=True)
                    break
        nMut = case_muts['chrom'].isna().sum()
    
    case_muts = case_muts.astype({"pos":int})
    return(case_muts)

def simulate_mut_vafs(tumor, rank, n) -> list: 

    """
    A function to simulate the VAF of each mutation
    """
    
    prop_vaf_file:pd.DataFrame = pd.read_csv(f"/genomeGAN/trained_models/vaf_annotation_by_study.tsv", sep='\t')
    prop_vaf_file = prop_vaf_file.loc[prop_vaf_file["study"]==tumor, ['vaf_range', rank]]
    mut_vafs:list = random.choices(list(prop_vaf_file['vaf_range']), weights=list(prop_vaf_file[rank]), k=n)
    mut_vafs = vaf_rank2float(mut_vafs)

    return(mut_vafs)

def chrom2int(chrom) -> int:

    """
    Convert the chromosome to an integer
    """

    if chrom.isdigit():
        return int(chrom)
    elif chrom == 'X':
        return 23
    elif chrom == 'Y':
        return 24
    else:
        return chrom
    
def chrom2str(chrom) -> str:

    """
    Convert the chromosome to a string
    """

    if chrom == 23:
        return 'X'
    elif chrom == 24:
        return 'Y'
    else:
        return str(chrom)

def assign_drivers(vcf, drivers_counts, drivers_mutations, drivers_vaf, drivers_tumor, fasta, donorID) -> pd.DataFrame:

    """
    Include driver mutations in the vcf with passenger mutations
    """

    # Select drivers from PCAWG donors
    drivers_counts = drivers_counts[drivers_counts != 0]
    selected_drivers:pd.DataFrame = pd.DataFrame()
    for driver, n in drivers_counts.items():
        if drivers_tumor == 'Lymph-MCLL':
            drivers_mutations_df = drivers_mutations['MUT']['mutations']
            driver_rows:pd.DataFrame = drivers_mutations_df[drivers_mutations_df['driver'] == driver]
        elif drivers_tumor == 'Lymph-UCLL':
            drivers_mutations_df = drivers_mutations['UNMUT']['mutations']
            driver_rows:pd.DataFrame = drivers_mutations_df[drivers_mutations_df['driver'] == driver]
        else:
            drivers_mutations_df = drivers_mutations['mutations']
            driver_rows:pd.DataFrame = drivers_mutations_df[drivers_mutations_df['driver'] == driver]
        selected_drivers = pd.concat([selected_drivers, driver_rows.sample(n=n, replace=False)], ignore_index=True)

    # Set chrom column to str
    selected_drivers['chrom'] = selected_drivers['chrom'].astype(str)

    # Fix indels ref and alt
    for _,mut in selected_drivers.iterrows():
        if mut['mut'] == 'DEL':
            mut['start'] = mut['start'] - 1
            prev_base:str = fasta[str(mut['chrom'])][mut['start']-1:mut['start']].seq
            mut['ref'] = f"{prev_base}{mut['ref']}"
            mut['alt'] = prev_base
        elif mut['mut'] == 'INS':
            base:str = fasta[mut['chrom']][mut['start']-1:mut['start']].seq
            mut['ref'] = base
            mut['alt'] = f"{base}{mut['alt']}"
        else:
            pass

    # Reorganize columns
    selected_drivers['VAF'] = drivers_vaf
    selected_drivers['ID'] = [f"sim{donorID+1}"] * drivers_counts.sum()
    selected_drivers['driver'] = selected_drivers['driver'].apply(lambda x: f'driver_{x}')
    selected_drivers.rename(columns={'chrom':'#CHROM', 'start':'POS', 'ref':'REF', 'alt':'ALT', 'driver':'MUT'}, inplace=True)
    selected_drivers = selected_drivers[['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'VAF', 'MUT']]

    # Concatenate
    vcf = pd.concat([vcf, selected_drivers], ignore_index=True)

    return(vcf)

def pd2vcf(muts, drivers_counts, drivers_mutations, drivers_vaf, drivers_tumor, fasta, donorID) -> pd.DataFrame:

    """
    Convert the pandas DataFrame into a VCF
    """

    # Create the appropiate row for each type of mutation
    new_ref_list:list = []
    new_alt_list:list = []
    for _,mut in muts.iterrows():
        chrom,pos,ref,alt,mut_len,mut_type = str(mut['chrom']),int(mut['pos']),str(mut['r.ctx']),str(mut['a.ctx']),int(mut['len']),str(mut['mut'])
        if mut_type == "SNP":
            new_ref_list.append(ref[1])
            new_alt_list.append(alt[1])
            continue
        elif mut_type == "DNP":
            new_ref_list.append("".join(ref[1:]))
            new_alt_list.append("".join(alt[1:]))
            continue
        elif mut_type == "TNP":
            new_ref_list.append(ref)
            new_alt_list.append(alt)
            continue
        elif mut_type == "INS":
            new_ref_list.append(ref[1])
            nt = ["A", "C", "G", "T"]
            ## Simulate a random insertion considering the simulated length
            new_alt_list.append(ref[1]+"".join(random.choices(nt, k = mut_len)))
            continue
        elif mut_type == "DEL":
            ## Get the region that have to be deleted from fasta
            ctx = fasta[chrom][pos:pos+abs(mut_len)].seq
            new_ref_list.append(ref[1]+ctx)
            new_alt_list.append(ref[1])
            continue
    
    # Create the VCF
    vcf:dict = {"#CHROM":muts['chrom'],
                    "POS":muts['pos'],
                    "ID":[f"sim{donorID+1}"] * len(muts['chrom']),
                    "REF":new_ref_list,
                    "ALT":new_alt_list,
                    "VAF":muts['vaf'],
                    "MUT":muts['mut'],
                    "SIGNATURE":muts['signature']}
    vcf = pd.DataFrame(vcf)
    vcf["MUT"][vcf["MUT"] == "SNP"] = vcf["SIGNATURE"][vcf["MUT"] == "SNP"]
    vcf.drop('SIGNATURE', axis=1, inplace=True)

    # Assign driver mutations to the VCF
    if drivers_counts.sum() > 0:
        vcf = assign_drivers(vcf, drivers_counts, drivers_mutations, drivers_vaf, drivers_tumor, fasta, donorID)

    # Sort the VCF
    vcf['#CHROM'] = vcf['#CHROM'].apply(chrom2int)
    vcf = vcf.sort_values(by=['#CHROM', 'POS'])
    vcf['#CHROM'] = vcf['#CHROM'].apply(chrom2str)

    return(vcf)

@click.group()
def cli():
    pass

@click.command(name="availTumors")
def availTumors():
    
    """
    List of available tumors to simulate
    """

    tumors:list = [["CNS-PiloAstro", "Liver-HCC", "Lymph-CLL"]]
    tumors:str = '\n'.join(['\t\t'.join(x) for x in tumors])
    click.echo(f"\nThis is the list of available tumor types that can be simulated using genomeGAN:\n\n{tumors}\n")

@click.command(name="vcfGANerator")
@click.option("-@", "--cpus",
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of CPUs to use")
@click.option("--tumor",
              type=click.Choice(["CNS-PiloAstro","Liver-HCC","Lymph-CLL"]),
              metavar="TEXT",
              show_choices=False,
              required = True,
              help="Tumor type to be simulated. Run 'availTumors' subcommand to check the list of available tumors that can be simulated")
@click.option("-n", "--nCases", "nCases",
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of cases to simulate")
@click.option("-r", "--refGenome", "refGenome",
              type=click.Path(exists=True, file_okay=True),
              required = True,
              help="hg19 reference genome in fasta format")
@click.option("--prefix",
              type=click.STRING,
              help="Prefix to name the output. If not, '--tumor' option is used as prefix")
@click.option("--outDir", "outDir",
              type=click.Path(exists=False, file_okay=False),
              default=os.getcwd(),
              show_default=False,
              help="Directory where save the simulations. Default is the current directory")
def genomeGAN(cpus, tumor, nCases, refGenome, prefix, outDir):

    """
    Command to simulate mutations (VCF) for different tumor types using a GAN model
    """
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # Torch options
    device:str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get models
    countModel, mutModel, posModel, driversModel, countCorr, countEx = tumor_models(tumor, device)

    # Load reference genome
    fasta = Fasta(refGenome)

    # Generate the counts for each type of mutation for each case
    counts:pd.DataFrame = simulate_counts(tumor, countModel, nCases, countCorr, countEx)

    # Annotate VAF rank to each donor
    donors_vafRank:list = simulate_vaf_rank(tumor, nCases)

    # Simulate one donor at a time
    muts:pd.DataFrame = pd.DataFrame()
    for idx in tqdm(range(nCases), desc = "Donors"):
        # Focus in one case
        case_counts:pd.Series = counts.iloc[idx]
        nMut:int = int(case_counts.sum())
        case_rank:str = donors_vafRank[idx]

        # Detect specific tumor type in case we are simulating Lymph-CLL
        if tumor == 'Lymph-CLL':
            drivers_tumor:str = 'Lymph-MCLL' if case_counts['SBS9'] > 0 else 'Lymph-UCLL'
        else:
            drivers_tumor:str = tumor

        # Simulate driver mutations to each donor
        case_drivers:pd.Series = simulate_drivers(drivers_tumor, driversModel)

        # Gender selection
        gender:str = gender_selection(tumor)

        # Generate the mutations
        muts, case_muts = simulate_mutations(mutModel, muts, nMut, case_counts, drivers_tumor)
        
        # Select the mutations corresponding for this case
        muts, case_muts = select_case_mutations(muts, case_counts)

        # Reduce muts size over rounds
        if muts.shape[0] > 1e7:
            muts = pd.DataFrame()
        else:
            muts = muts.sample(frac=0.5).reset_index(drop=True)

        # Generate the chromosome and position of the mutations
        case_muts = assign_position(tumor, case_counts, case_muts, posModel, nMut, fasta, gender, cpus)

        # Generate and assign the VAF to the mutations
        mut_vafs:list = simulate_mut_vafs(tumor, case_rank, nMut) #TODO - Adjust VAF for sexual mutations
        case_muts['vaf'] = mut_vafs
        drivers_vafs:list = simulate_mut_vafs(tumor, case_rank, case_drivers.sum())

        # Create the VCF output
        vcf = pd2vcf(case_muts, case_drivers, driversModel, drivers_vafs, drivers_tumor, fasta, idx)

        # Write the VCF
        output:str = out_path(outDir, prefix, tumor, idx+1)
        with open(output, "w+") as out:
            out.write("##fileformat=VCFv4.2\n")
        vcf.to_csv(output, sep="\t", index=False, mode="a")

cli.add_command(availTumors)
cli.add_command(genomeGAN)
if __name__ == '__main__':
    cli()