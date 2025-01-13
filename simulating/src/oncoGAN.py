#!/usr/local/bin/python3

import sys
sys.path.append('/oncoGAN/')

import os
import re
import click
import pickle
import torch
import random
import pandas as pd
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from liftover import ChainFile
from tqdm import tqdm
from pyfaidx import Fasta

VERSION = "0.2"

def tumor_models(tumor, device) -> list:

    """
    Get the specific models for the selected tumor type
    """

    # Counts model
    if tumor == "Breast-AdenoCa":
        countModel:dict = {}
        countModel['x1'] = torch.load(f"/oncoGAN/trained_models/counts/Breast-AdenoCa1_counts.pkl", map_location=device)
        countModel['x2'] = torch.load(f"/oncoGAN/trained_models/counts/Breast-AdenoCa2_counts.pkl", map_location=device)
    elif tumor == "CNS-PiloAstro":
        countModel:dict = {}
        countModel['x1'] = torch.load(f"/oncoGAN/trained_models/counts/CNS-PiloAstro1_counts.pkl", map_location=device)
        countModel['x2'] = torch.load(f"/oncoGAN/trained_models/counts/CNS-PiloAstro2_counts.pkl", map_location=device)
    elif tumor == "Eso-AdenoCa":
        countModel:dict = {}
        countModel['x1'] = torch.load(f"/oncoGAN/trained_models/counts/Eso-AdenoCa1_counts.pkl", map_location=device)
        countModel['x2'] = torch.load(f"/oncoGAN/trained_models/counts/Eso-AdenoCa2_counts.pkl", map_location=device)
        countModel['x3'] = torch.load(f"/oncoGAN/trained_models/counts/Eso-AdenoCa3_counts.pkl", map_location=device)
    elif tumor == "Kidney-RCC":
        countModel = torch.load(f"/oncoGAN/trained_models/counts/Kidney-RCC_counts.pkl", map_location=device)
    elif tumor == "Liver-HCC":
        countModel:dict = {}
        countModel['x1'] = torch.load(f"/oncoGAN/trained_models/counts/Liver-HCC1_counts.pkl", map_location=device)
        countModel['x2'] = torch.load(f"/oncoGAN/trained_models/counts/Liver-HCC2_counts.pkl", map_location=device)
    elif tumor == "Lymph-CLL":
        countModel:dict = {}
        countModel['MUT']:dict = {}
        countModel['MUT']['x1'] = torch.load(f"/oncoGAN/trained_models/counts/Lymph-MCLL1_counts.pkl", map_location=device)
        countModel['MUT']['x2'] = torch.load(f"/oncoGAN/trained_models/counts/Lymph-MCLL2_counts.pkl", map_location=device)
        countModel['UNMUT']:dict = {}
        countModel['UNMUT']['x1'] = torch.load(f"/oncoGAN/trained_models/counts/Lymph-UCLL1_counts.pkl", map_location=device)
        countModel['UNMUT']['x2'] = torch.load(f"/oncoGAN/trained_models/counts/Lymph-UCLL2_counts.pkl", map_location=device)
    elif tumor == 'Panc-Endocrine':
        countModel:dict = {}
        countModel['x1'] = torch.load(f"/oncoGAN/trained_models/counts/Panc-Endocrine1_counts.pkl", map_location=device)
        countModel['x2'] = torch.load(f"/oncoGAN/trained_models/counts/Panc-Endocrine2_counts.pkl", map_location=device)
    elif tumor == "Prost-AdenoCA":
        countModel = torch.load(f"/oncoGAN/trained_models/counts/Prost-AdenoCA_counts.pkl", map_location=device)
    
    # Mutations model
    if tumor == "Lymph-CLL":
        mutModel:dict = {}
        mutModel['MUT'] = torch.load(f"/oncoGAN/trained_models/mutations/Lymph-CLL_mutations.pkl", map_location=device)
        mutModel['UNMUT'] = torch.load(f"/oncoGAN/trained_models/mutations/Lymph-CLL_mutations.pkl", map_location=device)
    else:
        mutModel = torch.load(f"/oncoGAN/trained_models/mutations/{tumor}_mutations.pkl", map_location=device)

    # Drivers model and files
    if tumor == "Lymph-CLL":
        driversModel:dict = {}
        driversModel['MUT']:dict = {}
        driversModel['MUT']['model'] = torch.load(f"/oncoGAN/trained_models/drivers/Lymph-MCLL_drivers.pkl", map_location=device)
        driversModel['MUT']['mutations'] = pd.read_csv(f"/oncoGAN/trained_models/drivers/Lymph-MCLL_driver_mutations.csv")
        driversModel['UNMUT']:dict = {}
        driversModel['UNMUT']['model']:dict = {}
        driversModel['UNMUT']['model']['x1'] = torch.load(f"/oncoGAN/trained_models/drivers/Lymph-UCLL1_drivers.pkl", map_location=device)
        driversModel['UNMUT']['model']['x2'] = torch.load(f"/oncoGAN/trained_models/drivers/Lymph-UCLL2_drivers.pkl", map_location=device)
        driversModel['UNMUT']['model']['x3'] = torch.load(f"/oncoGAN/trained_models/drivers/Lymph-UCLL3_drivers.pkl", map_location=device)
        driversModel['UNMUT']['mutations'] = pd.read_csv(f"/oncoGAN/trained_models/drivers/Lymph-UCLL_driver_mutations.csv")
    elif tumor == "Panc-Endocrine":
        driversModel:dict = {}
        driversModel['model']:dict = {}
        driversModel['model']['x1'] = torch.load(f"/oncoGAN/trained_models/drivers/Panc-Endocrine1_drivers.pkl", map_location=device)
        driversModel['model']['x2'] = torch.load(f"/oncoGAN/trained_models/drivers/Panc-Endocrine2_drivers.pkl", map_location=device)
        driversModel['mutations'] = pd.read_csv(f"/oncoGAN/trained_models/drivers/{tumor}_driver_mutations.csv")
    else:
        driversModel:dict = {}
        driversModel['model'] = torch.load(f"/oncoGAN/trained_models/drivers/{tumor}_drivers.pkl", map_location=device)
        driversModel['mutations'] = pd.read_csv(f"/oncoGAN/trained_models/drivers/{tumor}_driver_mutations.csv")

    # Positions model
    if tumor == "Lymph-CLL":
        posModel:dict = {}
        with open(f"/oncoGAN/trained_models/positions/Lymph-MCLL_positions.pkl", 'rb') as f:
           posModel['MUT'] = pickle.load(f)
        with open(f"/oncoGAN/trained_models/positions/Lymph-UCLL_positions.pkl", 'rb') as f:
           posModel['UNMUT'] = pickle.load(f)
    else:
        with open(f"/oncoGAN/trained_models/positions/{tumor}_positions.pkl", 'rb') as f:
           posModel = pickle.load(f)

    # Counts corrections
    countsCorr:pd.DataFrame = pd.read_csv(f"/oncoGAN/trained_models/counts/counts_correction_rates.csv")

    # Counts exclusions
    countsEx:pd.DataFrame = pd.read_csv(f"/oncoGAN/trained_models/counts/counts_exclusions.csv")

    return(countModel, mutModel, posModel, driversModel, countsCorr, countsEx)

def cna_sv_models(device) -> list:

    """
    Get the CNA and SV models
    """
    
    cna_sv_countModel = torch.load("/oncoGAN/trained_models/cna_sv/CNA_SV_counts.pkl", map_location=device)
    cnaModel = torch.load("/oncoGAN/trained_models/cna_sv/CNA_model.pkl", map_location=device)
    svModel:dict = {}
    with open("/oncoGAN/trained_models/cna_sv/SV_positions.pkl", 'rb') as f:
        svModel['pos'] = pickle.load(f)
    svModel['sv'] = torch.load("/oncoGAN/trained_models/cna_sv/SV_model.pkl", map_location=device)

    return(cna_sv_countModel, cnaModel, svModel)

def out_path(outDir, prefix, tumor, n) -> click.Path:

    """
    Get the absolute path and name for the outputs
    """

    if prefix is not None:
        output:click.Path = f"{outDir}/{prefix}_sim{n}.vcf"
    else:
        output:click.Path = f"{outDir}/{tumor}_sim{n}.vcf"
    
    return(output)

def preprocess_counts(counts, tumor, corrections, exclusions) -> pd.DataFrame:
    
    """
    Function to preprocess the counts
    """

    # Select corrections and exclusions tumor rows
    corrections = corrections.loc[corrections["tumor"]==tumor]
    corrections = corrections.drop('tumor', axis=1).reset_index(drop=True)
    exclusions = exclusions.loc[exclusions["tumor"]==tumor]
    exclusions = exclusions.drop('tumor', axis=1).reset_index(drop=True)

    # Assign a donor index
    counts = counts.astype(float)
    counts['donor'] = [i for i in range(counts.shape[0])]
    
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
    counts = counts.dropna(axis=0, how='any').round(0).astype(int).reset_index(drop=True)
    
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
        counts:pd.DataFrame = pd.DataFrame()
        while counts.shape[0] < nCases:
            # Model 1
            x1_counts:pd.DataFrame = pd.DataFrame()
            while x1_counts.shape[0] < nCases:
                tmp_counts:pd.DataFrame = pd.DataFrame()
                for _ in range(5):
                    tmp:pd.DataFrame = countSynthesizer['x1'].generate_samples(nCases)
                    tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
                tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
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
            x1_counts = x1_counts.drop(['keep', 'keep2', 'total'], axis=1).reset_index(drop=True)

            # Model 2
            x2_counts:pd.DataFrame = pd.DataFrame()
            while x2_counts.shape[0] < nCases:
                tmp_counts:pd.DataFrame = pd.DataFrame()
                for _ in range(5):
                    tmp:pd.DataFrame = countSynthesizer['x2'].generate_samples(nCases)
                    tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
                tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
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
            x2_counts = x2_counts.drop(['keep', 'keep2', 'total'], axis=1).reset_index(drop=True)
            
            # Merge
            counts:pd.DataFrame = pd.concat([counts, x1_counts, x2_counts], ignore_index=True)
            ## General filters
            counts['total'] = counts.sum(axis=1)
            counts['modify'] = np.random.choice([True, False], size=counts.shape[0], p=[0.7, 0.3])
            counts['SBS2'] = np.where(counts['modify'], counts['SBS2'], 0)
            counts['SBS13'] = np.where(counts['modify'], counts['SBS13'], 0)
            counts['SBS1'] = counts.apply(lambda x: 
                                          np.where((x['SBS1'] > 300) & (x['SBS1'] < 450),
                                                   np.random.choice([x['SBS1']*np.random.choice([x/100 for x in range(60, 101, 5)]), x['SBS1']], p=[0.7, 0.3]),
                                                   x['SBS1']),
                                                axis=1)
            counts['SBS2'] = counts.apply(lambda x: 
                                          np.where((x['SBS2']/x['total'] > 0.12) & (x['SBS2']/x['total'] < 0.2),
                                                   np.random.choice([x['SBS2']*np.random.choice([x/100 for x in range(25, 40, 5)]), x['SBS2']], p=[0.8, 0.2]),
                                                   x['SBS2']),
                                                axis=1)
            counts['keep'] = counts.apply(lambda x:
                                          np.where(x['SBS8'] == 0,
                                                   np.random.choice([True, False], p=[0.8, 0.2]),
                                                   True),
                                                axis=1)
            counts = counts.loc[counts["keep"]==True]
            counts = counts.drop(columns=['modify', 'total', 'keep'])

            # Be sure donors' counts meet requirements
            i:int = 0
            while True:
                initial_shape:int = counts.shape[0]
                counts = preprocess_counts(counts, tumor, corrections, exclusions)
                end_shape:int = counts.shape[0]
                i += 1
                if (initial_shape == end_shape) | (i > 10):
                    break

        # Return counts
        counts = counts.sample(n=nCases).reset_index(drop=True)
        counts.fillna(0, inplace=True)
        counts = counts.round(0).astype(int)
        return(counts)
    
    elif tumor == "CNS-PiloAstro":
        counts:pd.DataFrame = pd.DataFrame()
        while counts.shape[0] < nCases:
            # Model 1
            x1_counts:pd.DataFrame = pd.DataFrame()
            while x1_counts.shape[0] < nCases:
                tmp_counts:pd.DataFrame = pd.DataFrame()
                for _ in range(5):
                    tmp:pd.DataFrame = countSynthesizer['x1'].generate_samples(nCases)
                    tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
                tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
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
                tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
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
            counts:pd.DataFrame = pd.concat([counts, x1_counts, x2_counts], ignore_index=True)

            # Be sure donors' counts meet requirements
            i:int = 0
            while True:
                initial_shape:int = counts.shape[0]
                counts = preprocess_counts(counts, tumor, corrections, exclusions)
                end_shape:int = counts.shape[0]
                i += 1
                if (initial_shape == end_shape) | (i > 10):
                    break

        # Return counts
        counts = counts.sample(n=nCases).reset_index(drop=True)
        counts.fillna(0, inplace=True)
        counts = counts.round(0).astype(int)
        return(counts)
    
    elif tumor == "Eso-AdenoCa":
        counts:pd.DataFrame = pd.DataFrame()
        while counts.shape[0] < nCases:
            # Model 1
            x1_counts:pd.DataFrame = pd.DataFrame()
            while x1_counts.shape[0] < nCases:
                tmp_counts:pd.DataFrame = pd.DataFrame()
                for _ in range(5):
                    tmp:pd.DataFrame = countSynthesizer['x1'].generate_samples(nCases)
                    tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
                tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
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
                tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
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
                tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
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
            counts:pd.DataFrame = pd.concat([counts, x1_counts, x2_counts, x3_counts], ignore_index=True)
            ## General filters
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

            # Be sure donors' counts meet requirements
            i:int = 0
            while True:
                initial_shape:int = counts.shape[0]
                counts = preprocess_counts(counts, tumor, corrections, exclusions)
                end_shape:int = counts.shape[0]
                i += 1
                if (initial_shape == end_shape) | (i > 10):
                    break

        # Return counts
        counts = counts.sample(n=nCases).reset_index(drop=True)
        counts.fillna(0, inplace=True)
        counts = counts.round(0).astype(int)
        return(counts)
    
    elif tumor == "Kidney-RCC":
        counts:pd.DataFrame = pd.DataFrame()
        while counts.shape[0] <= nCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer.generate_samples(nCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
            counts = pd.concat([counts,tmp_counts], ignore_index=True)

            # Be sure donors' counts meet requirements
            i:int = 0
            while True:
                initial_shape:int = counts.shape[0]
                counts = preprocess_counts(counts, tumor, corrections, exclusions)
                end_shape:int = counts.shape[0]
                i += 1
                if (initial_shape == end_shape) | (i > 10):
                    break

        # Return counts
        counts = counts.sample(n=nCases).reset_index(drop=True)
        counts.fillna(0, inplace=True)
        counts = counts.round(0).astype(int)
        return(counts)

    elif tumor == "Liver-HCC":
        counts:pd.DataFrame = pd.DataFrame()
        while counts.shape[0] < nCases:
            # Model 1
            x1_counts:pd.DataFrame = pd.DataFrame()
            while x1_counts.shape[0] < nCases:
                tmp_counts:pd.DataFrame = pd.DataFrame()
                for _ in range(5):
                    tmp:pd.DataFrame = countSynthesizer['x1'].generate_samples(nCases)
                    tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
                tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
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
                tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
                x2_counts = pd.concat([x2_counts,tmp_counts], ignore_index=True)
            ## Specific model filters
            x2_counts = x2_counts.loc[x2_counts["SBS8"]!=0]
            
            # Merge
            counts:pd.DataFrame = pd.concat([counts, x1_counts, x2_counts], ignore_index=True)

            # Be sure donors' counts meet requirements
            i:int = 0
            while True:
                initial_shape:int = counts.shape[0]
                counts = preprocess_counts(counts, tumor, corrections, exclusions)
                end_shape:int = counts.shape[0]
                i += 1
                if (initial_shape == end_shape) | (i > 10):
                    break

        # Return counts
        counts = counts.sample(n=nCases).reset_index(drop=True)
        counts.fillna(0, inplace=True)
        counts = counts.round(0).astype(int)
        return(counts)
    
    elif tumor == "Lymph-CLL":
        mCases:int = round(nCases*0.42)
        uCases:int = nCases - mCases

        # MCLL
        m_counts:pd.DataFrame = pd.DataFrame()
        while m_counts.shape[0] < nCases:
            ## Model 1
            x1m_counts:pd.DataFrame = pd.DataFrame()
            while x1m_counts.shape[0] < mCases:
                tmp_counts:pd.DataFrame = pd.DataFrame()
                for _ in range(5):
                    tmp:pd.DataFrame = countSynthesizer['MUT']['x1'].generate_samples(mCases)
                    tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
                tmp_counts = preprocess_counts(tmp_counts, 'Lymph-MCLL', corrections, exclusions)
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
                tmp_counts = preprocess_counts(tmp_counts, 'Lymph-MCLL', corrections, exclusions)
                x2m_counts = pd.concat([x2m_counts,tmp_counts], ignore_index=True)
            ### Specific model filters
            x2m_counts = x2m_counts.sample(n=x2m_n_keep)

            ## Merge
            m_counts:pd.DataFrame = pd.concat([m_counts, x1m_counts, x2m_counts], ignore_index=True)

            # Be sure donors' counts meet requirements
            i:int = 0
            while True:
                initial_shape:int = m_counts.shape[0]
                m_counts = preprocess_counts(m_counts, tumor, corrections, exclusions)
                end_shape:int = m_counts.shape[0]
                i += 1
                if (initial_shape == end_shape) | (i > 10):
                    break
        
        # Return MCLL counts
        m_counts = m_counts.sample(n=mCases).reset_index(drop=True)
        m_counts.fillna(0, inplace=True)

        # UCLL
        u_counts:pd.DataFrame = pd.DataFrame()
        while u_counts.shape[0] < nCases:
            ## Model 1
            x1u_counts:pd.DataFrame = pd.DataFrame()
            while x1u_counts.shape[0] < uCases:
                tmp_counts:pd.DataFrame = pd.DataFrame()
                for _ in range(5):
                    tmp:pd.DataFrame = countSynthesizer['UNMUT']['x1'].generate_samples(uCases)
                    tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
                tmp_counts = preprocess_counts(tmp_counts, 'Lymph-UCLL', corrections, exclusions)
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
                tmp_counts = preprocess_counts(tmp_counts, 'Lymph-UCLL', corrections, exclusions)
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
            u_counts:pd.DataFrame = pd.concat([u_counts, x1u_counts, x2u_counts], ignore_index=True)

            # Be sure donors' counts meet requirements
            i:int = 0
            while True:
                initial_shape:int = u_counts.shape[0]
                u_counts = preprocess_counts(u_counts, tumor, corrections, exclusions)
                end_shape:int = u_counts.shape[0]
                i += 1
                if (initial_shape == end_shape) | (i > 10):
                    break
        
        # Return UCLL counts
        u_counts = u_counts.sample(n=uCases).reset_index(drop=True)
        u_counts.fillna(0, inplace=True)

        # Merge MCLL and UCLL
        counts:pd.DataFrame = pd.concat([m_counts, u_counts], ignore_index=True)
        counts = counts.sample(frac=1)
        counts.fillna(0, inplace=True)
        counts = counts.round(0).astype(int)

        # Return counts
        return(counts)
    
    elif tumor == "Panc-Endocrine":
        counts:pd.DataFrame = pd.DataFrame()
        while counts.shape[0] < nCases:
            # Model 1
            x1_counts:pd.DataFrame = pd.DataFrame()
            while x1_counts.shape[0] < nCases:
                tmp_counts:pd.DataFrame = pd.DataFrame()
                for _ in range(5):
                    tmp:pd.DataFrame = countSynthesizer['x1'].generate_samples(nCases)
                    tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
                tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
                x1_counts = pd.concat([x1_counts,tmp_counts], ignore_index=True)

            # Model 2
            x2_counts:pd.DataFrame = pd.DataFrame()
            while x2_counts.shape[0] < nCases:
                tmp_counts:pd.DataFrame = pd.DataFrame()
                for _ in range(5):
                    tmp:pd.DataFrame = countSynthesizer['x2'].generate_samples(nCases)
                    tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
                tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
                x2_counts = pd.concat([x2_counts,tmp_counts], ignore_index=True)
            ## Specific model filters
            x2_counts['total'] = x2_counts.sum(axis=1)
            x2_counts['keep'] = x2_counts.apply(lambda x: 
                                                np.where((x['SBS13'] / x['total'] < 0.1) & (x['SBS13'] / x['total'] > 0.02),
                                                        True,
                                                        False),
                                                axis=1)
            x2_counts = x2_counts.loc[x2_counts["keep"]==True]
            x2_counts['keep'] = x2_counts.apply(lambda x: 
                                                np.where(x['SBS2'] / x['total'] > 0.02,
                                                        np.random.choice([True, False], p=[0.3, 0.7]),
                                                        True),
                                                axis=1)
            x2_counts = x2_counts.loc[x2_counts["keep"]==True]
            x2_counts = x2_counts.drop(['keep', 'total'], axis=1).reset_index(drop=True)
            
            # Merge
            counts:pd.DataFrame = pd.concat([counts, x1_counts, x2_counts], ignore_index=True)
            ## General filters
            counts['total'] = counts.sum(axis=1)
            counts['SBS13'] = counts.apply(lambda x:
                                            np.where((x['SBS13'] / x['total'] > 0.1) & (x['SBS13'] / x['total'] < 0.25),
                                                    x['SBS13']/3,
                                                    x['SBS13']),
                                            axis=1)
            counts['SBS5'] = counts.apply(lambda x:
                                            np.where((x['SBS5'] / x['total'] < 0.5) & (x['total'] < 5000),
                                                    np.random.choice([x['SBS5']*2, x['SBS5']]),
                                                    x['SBS5']),
                                            axis=1)
            counts['keep'] = counts.apply(lambda x:
                                        np.where(((x['SBS13'] == 0) & (x['SBS2'] != 0)) | ((x['SBS13'] != 0) & (x['SBS2'] == 0)), 
                                                False,
                                                True),
                                        axis=1)
            counts = counts.loc[counts["keep"]==True]
            counts['keep'] = counts.apply(lambda x:
                                        np.where((x['SBS36']/x['total'] < 0.25) & (x['SBS36'] != 0) & (x['SBS2'] != 0), 
                                                False,
                                                True),
                                        axis=1)
            counts = counts.loc[counts["keep"]==True]
            counts['keep'] = counts.apply(lambda x:
                                        np.where((x['SBS5']/x['total'] > 0.7) & (x['SBS13'] != 0), 
                                                False,
                                                True),
                                        axis=1)
            counts = counts.loc[counts["keep"]==True]
            counts['keep'] = counts.apply(lambda x:
                                        np.where((x['SBS2'] == 0) & (x['SBS13'] == 0) & (x['SBS19'] == 0) & (x['SBS36'] == 0) & (x['SBS30'] != 0), 
                                                False,
                                                True),
                                        axis=1)
            counts = counts.loc[counts["keep"]==True]
            counts = counts.drop(['keep', 'total'], axis=1).reset_index(drop=True)

            # Be sure donors' counts meet requirements
            i:int = 0
            while True:
                initial_shape:int = counts.shape[0]
                counts = preprocess_counts(counts, tumor, corrections, exclusions)
                end_shape:int = counts.shape[0]
                i += 1
                if (initial_shape == end_shape) | (i > 10):
                    break

        # Return counts
        counts = counts.sample(n=nCases).reset_index(drop=True)
        counts.fillna(0, inplace=True)
        return(counts)

    elif tumor == "Prost-AdenoCA":
        counts:pd.DataFrame = pd.DataFrame()
        while counts.shape[0] < nCases:
            tmp_counts:pd.DataFrame = pd.DataFrame()
            for _ in range(5):
                tmp:pd.DataFrame = countSynthesizer.generate_samples(nCases)
                tmp_counts = pd.concat([tmp_counts,tmp], ignore_index=True)
            tmp_counts = preprocess_counts(tmp_counts, tumor, corrections, exclusions)
            counts = pd.concat([counts,tmp_counts], ignore_index=True)
            ## Specific model filters
            counts['total'] = counts.sum(axis=1)
            counts['keep'] = counts.apply(lambda x: 
                                                np.where((x['SBS8'] / x['total'] > 0.1) & (x['total'] > 20000) & (x['total'] < 60000),
                                                        False,
                                                        True),
                                                axis=1)
            counts = counts.loc[counts["keep"]==True]
            counts = counts.drop(['keep'], axis=1).reset_index(drop=True)
            ## General filters
            counts['keep'] = counts.apply(lambda x: 
                                                np.where((x['SBS1'] != 0) & (x['SBS37'] != 0) & (x['SBS1'] / x['total'] > 0.2),
                                                        False,
                                                        np.where((x['SBS37'] != 0) & (x['SBS40'] != 0) & ((x['SBS40'] / x['total'] > 0.55) | (x['SBS40'] / x['total'] < 0.25)),
                                                                False,
                                                                True)),
                                                axis=1)
            counts = counts.loc[counts["keep"]==True]
            counts = counts.drop(['total', 'keep'], axis=1).reset_index(drop=True)

            # Be sure donors' counts meet requirements
            i:int = 0
            while True:
                initial_shape:int = counts.shape[0]
                counts = preprocess_counts(counts, tumor, corrections, exclusions)
                end_shape:int = counts.shape[0]
                i += 1
                if (initial_shape == end_shape) | (i > 10):
                    break

        # Return counts
        counts = counts.sample(n=nCases).reset_index(drop=True)
        counts.fillna(0, inplace=True)
        counts = counts.round(0).astype(int)
        return(counts)
    
def simulate_vaf_rank(tumor, nCases) -> list:

    """
    Function to simulate the VAF range for each donor
    """

    rank_file:pd.DataFrame = pd.read_csv("/oncoGAN/trained_models/vaf_rank_list.tsv", sep='\t') 
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
        tmp_drivers = tmp_drivers.round(0).astype(int)
        tmp_drivers_CD36:pd.DataFrame = tmp_drivers[tmp_drivers['CD36_Intron'] != 1].reset_index(drop=True)
        tmp_drivers_CD36_1:pd.DataFrame = tmp_drivers[tmp_drivers['CD36_Intron'] == 1].sample(frac=0.4).reset_index(drop=True)
        drivers:pd.DataFrame = pd.concat([tmp_drivers_CD36,tmp_drivers_CD36_1], ignore_index=True)
    
    elif tumor == "Lymph-UCLL":
        x1_tmp_drivers:pd.DataFrame = driversSynthesizer['UNMUT']['model']['x1'].generate_samples(10)
        x2_tmp_drivers:pd.DataFrame = driversSynthesizer['UNMUT']['model']['x2'].generate_samples(10)
        x3_tmp_drivers:pd.DataFrame = driversSynthesizer['UNMUT']['model']['x3'].generate_samples(10)
        drivers:pd.DataFrame = pd.concat([x1_tmp_drivers, x2_tmp_drivers, x3_tmp_drivers], ignore_index=True)
   
    elif tumor == "Panc-Endocrine":
        x1_tmp_drivers:pd.DataFrame = driversSynthesizer['model']['x1'].generate_samples(10)
        x2_tmp_drivers:pd.DataFrame = driversSynthesizer['model']['x2'].generate_samples(10)
        drivers:pd.DataFrame = pd.concat([x1_tmp_drivers, x2_tmp_drivers], ignore_index=True)
        drivers = drivers.round(0).astype(int)
        drivers['PTEN_Coding'] = drivers.apply(lambda x: 
                                                        np.where(x['PTEN_Coding'] == 1,
                                                                    np.random.choice([x['PTEN_Coding'], 0]),
                                                                    x['PTEN_Coding']),
                                                        axis=1)
    
    elif tumor == "Prost-AdenoCA":
        drivers:pd.DataFrame = driversSynthesizer['model'].generate_samples(10)
        drivers = drivers.round(0).astype(int)
        drivers['FIP1L1_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['FIP1L1_Intron'] != 0,
                                                                 np.random.choice([x['FIP1L1_Intron'], 0], p=[0.7, 0.3]),
                                                                 x['FIP1L1_Intron']),
                                                        axis=1)
        drivers['MAD1L1_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['MAD1L1_Intron'] != 0,
                                                                 np.random.choice([x['MAD1L1_Intron'], 0], p=[0.7, 0.3]),
                                                                 x['MAD1L1_Intron']),
                                                        axis=1)
        drivers['SCAI_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['SCAI_Intron'] != 0,
                                                                 np.random.choice([x['SCAI_Intron'], 0], p=[0.7, 0.3]),
                                                                 x['SCAI_Intron']),
                                                        axis=1)
        drivers['ZFHX3_Intron'] = drivers.apply(lambda x: 
                                                        np.where(x['ZFHX3_Intron'] > 2,
                                                                 0,
                                                                 x['ZFHX3_Intron']),
                                                        axis=1)
    
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
    file = file[~(file['r.ctx'].str.contains('N') | file['a.ctx'].str.contains('N'))]
    file.drop(["start", "vaf", "r.ctx1", "r.ctx2", "r.ctx3", "a.ctx1", "a.ctx2", "a.ctx3"], axis=1, inplace=True)

    return(file)

def manually_simulate_dnp_tnp(mutSynthesizer, nMut, mut_type, tumor) -> pd.DataFrame:
    
        """
        Function to manually simulate DNP and TNP mutations
        """
    
        nt_dict = {'A':['C', 'G', 'T'], 'C':['A', 'G', 'T'], 'G':['A', 'C', 'T'], 'T':['A', 'C', 'G']}
        if tumor == "Lymph-MCLL":
            muts = mutSynthesizer['MUT'].generate_samples(int(nMut)*20)
        elif tumor == "Lymph-UCLL":
            muts = mutSynthesizer['UNMUT'].generate_samples(int(nMut)*20)
        else:
            muts = mutSynthesizer.generate_samples(int(nMut)*20)
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

def assign_chromosome(positions, cna=False, sv=False, gender=None) -> pd.DataFrame:
    
    """
    Function to assign a chromosome to each position
    """
    
    # Positions are encoded in a continuous scale, so we need to decode them
    if (cna or sv) and gender == 'F':
        position_decode:np.array  = np.array([0,249250621,492449994,690472424,881626700,1062541960,1233657027,1392795690,1539159712,1680373143,1815907890,1950914406,2084766301,2199936179,2307285719,2409817111,2500171864,2581367074,2659444322,2718573305,2781598825,2829728720,2881033286])
        max_length:int = 3036303846
        chromosome_decode:np.array  = np.array(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X'])
    else:
        position_decode:np.array  = np.array([0,249250621,492449994,690472424,881626700,1062541960,1233657027,1392795690,1539159712,1680373143,1815907890,1950914406,2084766301,2199936179,2307285719,2409817111,2500171864,2581367074,2659444322,2718573305,2781598825,2829728720,2881033286,3036303846])
        max_length:int = 3095677412
        chromosome_decode:np.array  = np.array(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y'])

    if not (cna or sv):
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
    elif cna:
        positions['pos'] = positions['len'].cumsum()
        pos_encoded:np.array  = np.digitize(positions["pos"], position_decode, right=True)
        ## Assign new columns
        positions["pos"] = positions["pos"] - np.take(position_decode, pos_encoded-1)
        positions["chrom"] = np.take(chromosome_decode, pos_encoded-1)
    elif sv:
        ## Extend chromosomes
        position_decode = np.concatenate((position_decode, position_decode+max_length))
        chromosome_decode = np.concatenate((chromosome_decode, chromosome_decode))

        start_encoded:np.array  = np.digitize(positions["start"], position_decode, right=True)
        end_encoded:np.array  = np.digitize(positions["end"], position_decode, right=True)
        ## Assign new columns
        positions["start"] = positions["start"] - np.take(position_decode, start_encoded-1)
        positions["end"] = positions["end"] - np.take(position_decode, end_encoded-1)
        positions["chrom1"] = np.take(chromosome_decode, start_encoded-1)
        positions["chrom2"] = np.take(chromosome_decode, end_encoded-1)
        positions.rename(columns={'start':'start1', 'end':'start2'}, inplace=True)

        ## Remove interchromosomal DEL, DUP, INV and intrachromosomal TRA
        positions['keep'] = False
        positions.loc[(positions['svclass'] != 'TRA') & (positions['chrom1'] == positions['chrom2']), 'keep'] = True
        positions.loc[(positions['svclass'] == 'TRA') & (positions['chrom1'] != positions['chrom2']), 'keep'] = True
        positions = positions[positions['keep']].drop(columns=['keep'])

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

    sexChrFile:pd.DataFrame = pd.read_csv("/oncoGAN/trained_models/xy_ranks.txt", sep='\t') 
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

def get_coordinates(posSynthesizer, nMut, sexChrom) -> pd.DataFrame:

    """
    Function to generate the coordinates of the mutations
    """

    # Generate the windows
    if sexChrom == 'X':
        x_ranks:list = ['[2.88e+07;2.91e+07)', '[2.91e+07;2.94e+07)' , '[2.94e+07;2.97e+07)', '[2.97e+07;3e+07)', '[3e+07;3.03e+07)', '[3.03e+07;3.06e+07)']
        step1:pd.DataFrame = pd.DataFrame()
        while step1.value_counts().sum() < nMut:
            tmp:pd.DataFrame = posSynthesizer['step1'].sample(num_rows = round(nMut*5))
            tmp = tmp[tmp['rank'].isin(x_ranks)]
            step1 = pd.concat([step1, tmp], ignore_index=True)
    elif sexChrom == 'Y':
        y_ranks:list = ['[3.03e+07;3.06e+07)', '[3.06e+07;3.09e+07)', '[3.09e+07;3.1e+07]']
        step1:pd.DataFrame = pd.DataFrame()
        while step1.value_counts().sum() < nMut:
            tmp:pd.DataFrame = posSynthesizer['step1'].sample(num_rows = round(nMut*5))
            tmp = tmp[tmp['rank'].isin(y_ranks)]
            step1 = pd.concat([step1, tmp], ignore_index=True)
    else:
        step1:pd.DataFrame = posSynthesizer['step1'].sample(num_rows = round(nMut*1.2))
    step1 = step1['rank'].value_counts()

    # Generate the specific position ranges
    positions:pd.DataFrame = pd.DataFrame()
    for rank, n in zip(step1.index, step1):
        try:
            positions = pd.concat([positions, posSynthesizer[rank].sample(num_rows = n)])
        except KeyError:
            continue
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
            tmp_positions:pd.DataFrame = get_coordinates(posSynthesizer, difference*10, None)
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
            tmp_positions:pd.DataFrame = get_coordinates(posSynthesizer, abs(difference)*100, sexChrom)
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
    positions:pd.DataFrame = get_coordinates(posSynthesizer, nMut, None)

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
    
    prop_vaf_file:pd.DataFrame = pd.read_csv(f"/oncoGAN/trained_models/vaf_annotation_by_study.tsv", sep='\t')
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
    selected_drivers['ID'] = [f"sim{donorID+1}"] * drivers_counts.sum()
    selected_drivers["QUAL"] = "."
    selected_drivers["FILTER"] = "."
    selected_drivers['driver'] = selected_drivers['driver'].apply(lambda x: f'driver_{x}')
    selected_drivers['INFO'] = selected_drivers.apply(lambda row: f"AF={drivers_vaf[row.name]};MS={row['driver']}", axis=1)
    selected_drivers.rename(columns={'chrom':'#CHROM', 'start':'POS', 'ref':'REF', 'alt':'ALT'}, inplace=True)
    selected_drivers = selected_drivers[['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']]

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
    mut_sig_list:list = []
    for _,mut in muts.iterrows():
        chrom,pos,ref,alt,mut_len,mut_type,signature = str(mut['chrom']),int(mut['pos']),str(mut['r.ctx']),str(mut['a.ctx']),int(mut['len']),str(mut['mut']),str(mut['signature'])
        if mut_type == "SNP":
            new_ref_list.append(ref[1])
            new_alt_list.append(alt[1])
            mut_sig_list.append(signature)
            continue
        elif mut_type == "DNP":
            new_ref_list.append("".join(ref[1:]))
            new_alt_list.append("".join(alt[1:]))
            mut_sig_list.append(mut_type)
            continue
        elif mut_type == "TNP":
            new_ref_list.append(ref)
            new_alt_list.append(alt)
            mut_sig_list.append(mut_type)
            continue
        elif mut_type == "INS":
            new_ref_list.append(ref[1])
            nt = ["A", "C", "G", "T"]
            ## Simulate a random insertion considering the simulated length
            new_alt_list.append(ref[1]+"".join(random.choices(nt, k = mut_len)))
            mut_sig_list.append(mut_type)
            continue
        elif mut_type == "DEL":
            ## Get the region that have to be deleted from fasta
            ctx = fasta[chrom][pos:pos+abs(mut_len)].seq
            new_ref_list.append(ref[1]+ctx)
            new_alt_list.append(ref[1])
            mut_sig_list.append(mut_type)
            continue
    
    # Create the VCF
    vcf:dict = {"#CHROM":muts['chrom'],
                "POS":muts['pos'],
                "ID":[f"sim{donorID+1}"] * len(muts['chrom']),
                "REF":new_ref_list,
                "ALT":new_alt_list,
                "QUAL":".",
                "FILTER":".",
                "INFO":[f"AF={vaf};MS={sig}" for vaf, sig in zip(muts['vaf'], mut_sig_list)]}
    vcf = pd.DataFrame(vcf)

    # Assign driver mutations to the VCF
    if drivers_counts.sum() > 0:
        vcf = assign_drivers(vcf, drivers_counts, drivers_mutations, drivers_vaf, drivers_tumor, fasta, donorID)

    # Sort the VCF
    vcf['#CHROM'] = vcf['#CHROM'].apply(chrom2int)
    vcf = vcf.sort_values(by=['#CHROM', 'POS'])
    vcf['#CHROM'] = vcf['#CHROM'].apply(chrom2str)

    # Filter out some random and very infrequent DNP, TNP and repeated SNPs
    vcf['keep'] = abs(vcf['POS'].diff()) > 2
    vcf = vcf[vcf['keep'].shift(-1, fill_value=False)]
    vcf = vcf.drop(columns=['keep']).reset_index(drop=True)

    return(vcf)

def hg19tohg38(vcf=None, cna=None, sv=None) -> pd.DataFrame:

    """
    Convert hg19 coordinates to hg38
    """

    if vcf is not None:
        converter = ChainFile('/.liftover/hg19ToHg38.over.chain.gz')
        for i,row in vcf.iterrows():
            chrom:str = str(row['#CHROM'])
            pos:int = int(row['POS'])
            try:
                liftOver_result:tuple = converter[chrom][pos][0]
                vcf.loc[i, '#CHROM'] = liftOver_result[0]
                vcf.loc[i, 'POS'] = liftOver_result[1]
            except IndexError:
                vcf.loc[i, '#CHROM'] = 'Remove'
        vcf = vcf[~vcf['#CHROM'].str.contains('Remove', na=False)]
        return(vcf)
    elif cna is not None:
        hg19_end:list = [249250621,492449994,690472424,881626700,1062541960,1233657027,1392795690,1539159712,1680373143,1815907890,1950914406,2084766301,2199936179,2307285719,2409817111,2500171864,2581367074,2659444322,2718573305,2781598825,2829728720,2881033286,3036303846,3095677412]
        hg38_end:list = [248956422,491149951,689445510,879660065,1061198324,1232004303,1391350276,1536488912,1674883629,1808681051,1943767673,2077042982,2191407310,2298451028,2400442217,2490780562,2574038003,2654411288,2713028904,2777473071,2824183054,2875001522,3031042417,3088269832]
        hg19_hg38_ends:dict = dict(zip(hg19_end, hg38_end))

        cna['end'] = cna['end'].apply(lambda x: hg19_hg38_ends.get(x, x))
        return(cna)
    elif sv is not None:
        chroms:list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']
        hg19_end:list = [249250621,492449994,690472424,881626700,1062541960,1233657027,1392795690,1539159712,1680373143,1815907890,1950914406,2084766301,2199936179,2307285719,2409817111,2500171864,2581367074,2659444322,2718573305,2781598825,2829728720,2881033286,3036303846,3095677412]
        hg38_end:list = [248956422,491149951,689445510,879660065,1061198324,1232004303,1391350276,1536488912,1674883629,1808681051,1943767673,2077042982,2191407310,2298451028,2400442217,2490780562,2574038003,2654411288,2713028904,2777473071,2824183054,2875001522,3031042417,3088269832]
        hg19_dict:dict = dict(zip(chroms, hg19_end))
        hg38_dict:dict = dict(zip(chroms, hg38_end))

        for i, row in sv.iterrows():
            ## Chrom1
            hg19_end1:int = hg19_dict.get(row['chrom1'])
            hg38_end1:int = hg38_dict.get(row['chrom1'])
            if row['end1'] > hg38_end1:
                sv.loc[i, 'end1'] = hg38_end1 - (hg19_end1 - row['end1'])
                sv.loc[i, 'start1'] = sv.loc[i, 'end1']-1

            ## Chrom2
            hg19_end2:int = hg19_dict.get(row['chrom2'])
            hg38_end2:int = hg38_dict.get(row['chrom2'])
            if row['end2'] > hg38_end2:
                sv.loc[i, 'end2'] = hg38_end2 - (hg19_end2 - row['end2'])
                sv.loc[i, 'start2'] = sv.loc[i, 'end2']-1
        return(sv)

def select_cna_sv_counts(cna_sv_countModel, nCases, tumor, counts) -> pd.DataFrame:

    """
    Generate CNA and SV for each donor
    """

    # Calculate the total mutations per donor
    counts_totalmut:pd.Series = counts.sum(axis=1)

    # Generate CNA and SV samples
    cna_sv_counts:pd.DataFrame = cna_sv_countModel.generate_samples(nCases*100, var_column='study', var_class=tumor)
    
    # Select CNA-SV events for each donor based on the total number of mutations as link between the two models
    selected_cna_sv_counts:pd.DataFrame = pd.DataFrame()
    for count in counts_totalmut:
        ## Find the index of the closest 'total_mut' in cna_sv_counts
        cna_sv_index:int = (np.abs(cna_sv_counts['total_mut'] - count)).argmin()
        ## Select the best case
        tmp:pd.DataFrame = cna_sv_counts.iloc[[cna_sv_index],:]
        selected_cna_sv_counts = pd.concat([selected_cna_sv_counts, tmp], ignore_index=True)
        ## Drop the selected row to avoid duplicate selections
        cna_sv_counts = cna_sv_counts.drop(index=cna_sv_index).reset_index(drop=True)

    return(selected_cna_sv_counts)

def select_cnas(cnas_df, nCNAs, lenCNA, iterations=100000) -> list:
    
    """
    Select a subset of CNAs such that the sum of the subset is as close as possible to the generated length.
    """
    
    best_sum_difference:float = float('inf')
    for _ in range(iterations):
        subset:pd.DataFrame = cnas_df.sample(nCNAs)
        subset_sum = subset['len'].sum()
        
        # Check if the current subset's sum is closer to the generated lenght
        if abs(subset_sum - lenCNA) < best_sum_difference:
            best_sum_difference = abs(subset_sum - lenCNA)
            best_subset = subset
        
        # Early exit if we reach a close subset
        if best_sum_difference < 0.1*lenCNA:
            break
    
    return(best_subset, int(np.sum(best_subset['len'])))

def sort_by_int_chrom(chrom) -> pd.DataFrame:
    
    """
    Sort a dataframe using integer chromosomes
    """

    if chrom == 'X':
        return 23
    if chrom == 'Y':
        return 24
    else:
        return int(chrom)

def rescue_missing_chroms(cnas_df, gender, keys, chrom_size_dict) -> pd.DataFrame:

    """
    A function to recover missing chromsomes for CNA
    """

    # Detect missing chromosomes
    all_chroms:set = set(cnas_df['chrom'].astype(str))
    simulated_chroms:list = [str(chrom) for chrom in keys if str(chrom) in all_chroms]
    simulated_indices:dict = {chrom: keys.index(chrom) for chrom in simulated_chroms}
    missing_chroms:list = [str(chrom) for chrom in keys if str(chrom) not in all_chroms]

    if gender == 'F' and 'Y' in missing_chroms:
        missing_chroms.remove('Y')

    # Select the next available chromosome as template
    for missing_chrom in missing_chroms:
        missing_index:int = keys.index(missing_chrom)
        remaining_simulated = [chrom for chrom, index in simulated_indices.items() if index > missing_index]
        if remaining_simulated:
            next_chrom:str = remaining_simulated[0]
        else:
            next_chrom:str = random.choice(simulated_chroms)
        next_chrom_row:pd.DataFrame = pd.DataFrame([cnas_df[cnas_df['chrom'] == next_chrom].iloc[0].copy()])
        next_chrom_row['chrom'] = missing_chrom
        next_chrom_row['pos'] = int(chrom_size_dict[missing_chrom])
        cnas_df = pd.concat([cnas_df, next_chrom_row], ignore_index=True)
    cnas_df = cnas_df.sort_values(by=['chrom', 'pos'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)

    return(cnas_df)

def adjust_cna_position(cnas_df, gender) -> pd.DataFrame:

    """
    Adjust the lengths of the CNAs to fit the assigned chromosomes positions
    """

    keys:list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']
    values:list = [249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540,102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560,59373566]
    chrom_size_dict:dict = dict(zip(keys, values))

    # Add missing chromsomes
    cnas_df = rescue_missing_chroms(cnas_df, gender, keys, chrom_size_dict)
    
    # Get the maximum position for each chrom
    grouped:pd.DataFrame = cnas_df.groupby('chrom')['pos'].max().reset_index()
    grouped.columns = ['chrom', 'max_pos']
    
    adjusted_cnas_df:pd.DataFrame = pd.DataFrame()
    for chrom, max_pos in grouped.itertuples(index=False):
        real_chrom_length:int = chrom_size_dict[chrom]
        ratio:float = real_chrom_length / max_pos
        
        # Adjust 'pos' for each chrom group and set max 'pos' value to real chrom length
        tmp_df:pd.DataFrame = cnas_df[cnas_df['chrom'] == chrom].copy()
        tmp_df['pos'] = (tmp_df['pos'] * ratio).round().astype(int)
        tmp_df['pos'][-1] = real_chrom_length
        adjusted_cnas_df = pd.concat([adjusted_cnas_df, tmp_df], ignore_index=True)
    
    adjusted_cnas_df = adjusted_cnas_df.sort_values(by=['chrom', 'pos'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)

    # Set the first start position of each chromsome as 1
    adjusted_cnas_df['start'] = adjusted_cnas_df.groupby('chrom')['pos'].transform(lambda x: np.insert(x.values + 1, 0, 1)[:-1])
    adjusted_cnas_df = adjusted_cnas_df.rename(columns={"pos": "end"})

    return(adjusted_cnas_df)

def combine_same_cna_events(cnas_df) -> pd.DataFrame:
    
    """
    Combines consecutive cnas with the same 'major_cn' and 'minor_cn' events
    """

    # Group by unique clusters of consecutive rows with the same chrom, major_cn, and minor_cn
    combined_df:pd.DataFrame = (
        cnas_df.groupby((cnas_df[['chrom', 'major_cn', 'minor_cn', 'donor_id', 'study']].shift() != cnas_df[['chrom', 'major_cn', 'minor_cn', 'donor_id', 'study']]).any(axis=1).cumsum())
          .agg(
              chrom=('chrom', 'first'),
              start=('start', 'min'),
              end=('end', 'max'),
              major_cn=('major_cn', 'first'),
              minor_cn=('minor_cn', 'first'),
              donor_id=('donor_id', 'first'),
              study=('study', 'first')
          )
          .reset_index(drop=True)
    )
    
    return(combined_df)

def simulate_cnas(nCNAs, lenCNA, tumor, cnaModel, gender, idx) -> pd.DataFrame:
    
    """
    Generate CNAs
    """

    max_length:int = 3036303846 if gender == "F" else 3095677412 if gender == "M" else None
    case_cnas:pd.DataFrame = cnaModel.generate_samples(nCNAs*100, var_column='study', var_class=tumor)

    # Update CNAs length
    lenCNA = round(np.exp(lenCNA))
    lenCNA_normal:int = max_length-lenCNA
    case_cnas['len'] = round(np.exp(case_cnas['len'])*10000)

    # Process altered haplotypes
    case_cnas_altered:pd.DataFrame = case_cnas[(case_cnas['major_cn'] != 1) | (case_cnas['minor_cn'] != 1)]
    selected_case_cnas_altered, selected_lenCNA = select_cnas(case_cnas_altered, nCNAs, lenCNA)
    len_adjust_ratio:float = lenCNA/selected_lenCNA
    selected_case_cnas_altered['len'] = round(selected_case_cnas_altered['len']*len_adjust_ratio).astype(int)

    # Process normal haplotypes
    case_cnas_normal:pd.DataFrame = case_cnas[(case_cnas['major_cn'] == 1) & (case_cnas['minor_cn'] == 1)]
    case_cnas_normal['cumsum'] = case_cnas_normal['len'].cumsum()
    selected_case_cnas_normal = case_cnas_normal[case_cnas_normal['cumsum'] <= lenCNA_normal]
    len_adjust_ratio_normal:float = lenCNA_normal/np.sum(selected_case_cnas_normal['len'])
    selected_case_cnas_normal['len'] = round(selected_case_cnas_normal['len']*len_adjust_ratio_normal).astype(int)
    selected_case_cnas_normal = selected_case_cnas_normal.drop(columns=['cumsum'])

    # Merge and shuffle CNAs
    case_cnas = pd.concat([selected_case_cnas_altered, selected_case_cnas_normal])
    case_cnas = case_cnas.sample(frac=1, replace=False, random_state=1).reset_index(drop=True)
    
    # Assign chromosomes and adjust CNAs by chromosome
    case_cnas = assign_chromosome(case_cnas, cna=True, gender=gender)
    case_cnas = adjust_cna_position(case_cnas, gender)
    
    # Add donor id
    case_cnas["donor_id"] = f"sim{idx}"
    case_cnas = case_cnas[["chrom", "start", "end", "major_cn", "minor_cn", "donor_id", "study"]]

    # Combine same CNAs events
    case_cnas = combine_same_cna_events(case_cnas)

    # Sort by real integer chrom order
    case_cnas = case_cnas.sort_values(by=['chrom', 'start'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)

    # Create an ID for each CNA segment
    case_cnas['id'] = 'cna' + case_cnas.index.astype(str)

    return(case_cnas)

def assign_cna_plot_color(y) -> str:

    """
    Asign a color to CNA segements depending on the copy number
    """
    
    if y == 1:
        return "Normal"
    elif y > 1:
        return "Gain"
    else:
        return "Loss"
    
def plot_cnas(cna_profile, sv_profile, tumor, output, idx) -> None:

    """
    Plot CNA segments
    """

    # Change chrom format to str
    cna_profile['chrom'] = cna_profile['chrom'].astype(str)
    sv_profile['chrom1'] = sv_profile['chrom1'].astype(str)
    sv_profile['chrom2'] = sv_profile['chrom2'].astype(str)

    # Define chromosome lengths
    chrom_list:list = list(range(1, 23)) + ["X", "Y"]
    cumlength_list:list = [0, 249250621, 492449994, 690472424, 881626700, 1062541960, 1233657027, 1392795690, 1539159712, 1680373143, 1815907890, 1950914406, 2084766301, 2199936179, 2307285719, 2409817111, 2500171864, 2581367074, 2659444322, 2718573305, 2781598825, 2829728720, 2881033286, 3036303846]
    cumlength_end_list:list = [249250621, 492449994, 690472424, 881626700, 1062541960, 1233657027, 1392795690, 1539159712, 1680373143, 1815907890, 1950914406, 2084766301, 2199936179, 2307285719, 2409817111, 2500171864, 2581367074, 2659444322, 2718573305, 2781598825, 2829728720, 2881033286, 3036303846, 3095677412]
    chrom_size_list:list = [end - start for start, end in zip(cumlength_list, cumlength_end_list)]
    chrom_cumsum_length:pd.DataFrame = pd.DataFrame({
        'chrom': [str(chr) for chr in chrom_list],
        'cumlength': cumlength_list,
        'cumlength_end': cumlength_end_list,
        'chrom_size': chrom_size_list
    })

    if 'Y' not in set(cna_profile['chrom']):
        chrom_cumsum_length = chrom_cumsum_length.iloc[:-1]

    # Preprocess the data
    ## Pivot longer SVs
    sv_profile['sv_id'] = ['sv{}'.format(i) for i in range(len(sv_profile))]
    ### Inversions
    sv_profile_inv = sv_profile[sv_profile['svclass'].isin(['h2hINV', 't2tINV'])]
    sv_profile_inv = sv_profile_inv.rename(columns={'chrom1': 'chrom', 'start1': 'start', 'start2': 'end'})
    sv_profile_inv = sv_profile_inv[['chrom', 'start', 'end', 'svclass', 'sv_id']]
    ### Translocations
    sv_profile_tra = sv_profile[sv_profile['svclass']=='TRA']
    sv_profile_long_tra = pd.concat([
        sv_profile_tra[['chrom1', 'start1', 'end1', 'svclass', 'sv_id']].rename(columns={'chrom1': 'chrom', 'start1': 'start', 'end1': 'end'}),
        sv_profile_tra[['chrom2', 'start2', 'end2', 'svclass', 'sv_id']].rename(columns={'chrom2': 'chrom', 'start2': 'start', 'end2': 'end'})])
    sv_profile_long_tra.reset_index(drop=True, inplace=True)
    ### Concatenate
    sv_profile_long = pd.concat([sv_profile_inv, sv_profile_long_tra])
    ## Left join CNAs and SVs with chromosome lengths
    cna_profile = cna_profile.merge(chrom_cumsum_length[['chrom', 'cumlength']], on='chrom', how='left')
    sv_profile_long = sv_profile_long.merge(chrom_cumsum_length[['chrom', 'cumlength']], on='chrom', how='left')
    ## Update segment positions
    cna_profile['start'] = cna_profile['start'] + cna_profile['cumlength']
    cna_profile['end'] = cna_profile['end'] + cna_profile['cumlength']
    sv_profile_long['start'] = sv_profile_long['start'] + sv_profile_long['cumlength']
    sv_profile_long['end'] = sv_profile_long['end'] + sv_profile_long['cumlength']
    ## Remove unnecesary columns
    cna_profile = cna_profile.drop(columns=['cumlength'])
    sv_profile_long = sv_profile_long.drop(columns=['cumlength'])
    ## Add a group column, one for each segment
    cna_profile['group'] = np.arange(1, len(cna_profile) + 1)
    sv_profile_long['group'] = np.arange(1, len(sv_profile_long) + 1)
    ## Calculate linewidth
    cna_profile['linewidth'] = np.where(cna_profile['major_cn'] == cna_profile['minor_cn'], 5, 3)
    ## Pivot longer 'major_cn' and 'minor_cn' columns for CNAs
    ### CNAs
    id_vars:list = [col for col in cna_profile.columns if col not in ['major_cn', 'minor_cn']]
    cna_profile_long = cna_profile.melt(
        id_vars=id_vars,
        value_vars=['major_cn', 'minor_cn'],
        var_name='cn',
        value_name='y'
    )
    ### SVs
    ymax = max(cna_profile_long['y'])
    sv_profile_long['overlap'] = sv_profile_long['start'] <= sv_profile_long['end'].shift()
    overlap = sv_profile_long['overlap'].tolist()
    for i in range(1, len(overlap)):
        if overlap[i] and overlap[i - 1]: 
            overlap[i] = False
    sv_profile_long['overlap'] = overlap
    sv_profile_long['y'] = sv_profile_long['svclass'].apply(lambda x: ymax+1.5 if x == 'TRA' else ymax+1) + sv_profile_long['overlap'].apply(lambda x: 0.2 if x else 0)
    sv_profile_long = sv_profile_long.drop(columns=['overlap'])
    ## Pivot longer 'start' and 'end' columns
    ### CNAs
    id_vars:list = [col for col in cna_profile_long.columns if col not in ['start', 'end']]
    cna_profile_long:pd.DataFrame = cna_profile_long.melt(
        id_vars=id_vars,
        value_vars=['start', 'end'],
        var_name='position',
        value_name='x'
    )
    cna_profile_long = cna_profile_long.sort_values('group').reset_index(drop=True)
    ### SVs
    id_vars:list = [col for col in sv_profile_long.columns if col not in ['start', 'end']]
    sv_profile_long_long:pd.DataFrame = sv_profile_long.melt(
        id_vars=id_vars,
        value_vars=['start', 'end'],
        var_name='position',
        value_name='x'
    )
    sv_profile_long_long = sv_profile_long_long.sort_values('group').reset_index(drop=True)

    # Plot
    ## Assign colors
    cna_profile_long['color'] = cna_profile_long['y'].apply(assign_cna_plot_color)
    cna_profile_long['color'] = pd.Categorical(
        cna_profile_long['color'],
        categories=['Gain', 'Normal', 'Loss'],
        ordered=True
    )
    cna_color_mapping:dict = {'Gain': "#2a9d8f", 'Normal': "#264653", 'Loss': "#f4a261"}
    sv_profile_long_long['svclass'] = pd.Categorical(
        sv_profile_long_long['svclass'],
        categories=['h2hINV', 't2tINV', 'TRA'],
        ordered=True
    )
    sv_color_mapping:dict = {'h2hINV': "#FF9898FF", 't2tINV': "#DC3262", 'TRA': "#7A0425"}
    ## Create a figure and axis object
    plt.figure(figsize=(16, 8))
    ## Plot the segments
    ### CNAs
    for (grp, cn), data in cna_profile_long.groupby(['group', 'cn']):
        plt.plot(
            data['x'], 
            data['y'], 
            color=cna_color_mapping[data['color'].iloc[0]], 
            linewidth=data['linewidth'].iloc[0],
            label='_nolegend_',
            solid_capstyle='butt'
        )
    ### SVs
    for grp, data in sv_profile_long_long.groupby(['group']):
        plt.plot(
            data['x'], 
            data['y'], 
            color=sv_color_mapping[data['svclass'].iloc[0]], 
            linewidth=4,
            label='_nolegend_',
            solid_capstyle='butt'
        )
    ## Separate chroms by vertical dashed lines
    chrom_vlines:pd.DataFrame = chrom_cumsum_length.iloc[:-1]
    for x in chrom_vlines['cumlength_end']:
        plt.axvline(x=x, color='gray', linewidth=0.2, linestyle='--')
    ## Calculate ymax for setting y-axis limits and label positions
    ymax:int = cna_profile_long['y'].max()
    plt.ylim(-0.5, ymax + 2.2)
    plt.yticks(range(0,ymax+1))
    ## Calculate chromosome label positions
    ### X
    chrom_cumsum_length['x_label_pos'] = chrom_cumsum_length['cumlength'] + chrom_cumsum_length['chrom_size'] / 2
    ### Y
    num_labels:int = len(chrom_cumsum_length)
    y_positions:list = [ymax + 2 if i % 2 == 0 else ymax + 2.3 for i in range(num_labels - 1)]
    y_positions.append(ymax + 2)  # Add the last position
    chrom_cumsum_length['y_label_pos'] = y_positions
    ## Add chromosome labels
    for _, row in chrom_cumsum_length.iterrows():
        plt.text(
            row['x_label_pos'], 
            row['y_label_pos'], 
            str(row['chrom']), 
            ha='center', 
            va='bottom', 
            fontsize=11
        )
    ## Minimal style
    sns.set_style("whitegrid", {'axes.grid': False})
    sns.despine(trim=True, left=False, bottom=False)
    ## Set axis labels and subtitle
    ax = plt.gca()
    plt.title(f'{tumor} - Donor{idx}', fontsize=16, y=1.1)
    plt.xlabel('Genome', fontsize=14)
    plt.tick_params(axis='x', which='both', length=0, labelbottom=False)
    plt.ylabel('CNA', fontsize=14)
    plt.tick_params(axis='y', which='both', length=5)
    loc, label = plt.yticks()
    ax.yaxis.set_label_coords(-0.02, np.mean(loc), transform=ax.get_yaxis_transform())
    ## Adjust legend position
    handles = [
        plt.Line2D([0], [0], color="#FF9898", lw=4, linestyle='-', label='h2hINV'),
        plt.Line2D([0], [0], color="#DC3262", lw=4, linestyle='-', label='t2tINV'),
        plt.Line2D([0], [0], color="#7A0425", lw=4, linestyle='-', label='TRA')
    ]
    plt.legend(
        handles=handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.1),
        ncol=3,
        fontsize=10,
        frameon=False
    )

    # Save the plot
    plt.savefig(output)

def get_sv_coordinates(n, svModel, gender) -> pd.DataFrame:
    
    """
    Generate genomic coordinates for SV
    """

    # Generate position ranks
    tmp_pos:pd.DataFrame = svModel['pos']['step1'].sample(num_rows = round(n*5))

    # Remove Y coordinates
    if gender == "F":
        y_ranks:list = ['[3.03e+07;3.06e+07)', '[3.06e+07;3.09e+07)', '[3.09e+07;3.1e+07]']
        tmp_pos = tmp_pos[~tmp_pos['rank'].isin(y_ranks)]
    
    # Keep the correct number of SV
    tmp_pos = tmp_pos.sample(n=n)

    # Simulate the exact positions
    step1:pd.Series = tmp_pos['rank'].value_counts()
    positions:pd.DataFrame = pd.DataFrame()
    for rank, m in zip(step1.index, step1):
        try:
            positions = pd.concat([positions, svModel['pos'][rank].sample(num_rows = m)])
        except KeyError:
            continue
    positions['start'] = positions['start']*100
    positions.reset_index(drop=True, inplace=True)

    return(positions['start'])

def check_sv_strand_patterns(sv_profile) -> pd.DataFrame:

    """
    Check SV strand patterns
    """

    expected_patterns:dict = {
        'DEL': ('+', '-'),
        'DUP': ('-', '+'),
        'h2hINV': ('+', '+'),
        't2tINV': ('-', '-'),
    }

    # Get expected strands for the svclass
    sv_profile['expected_strand1'], sv_profile['expected_strand2'] = zip(*sv_profile['svclass'].apply(lambda svclass: expected_patterns.get(svclass, (None, None))))

    # If the current strands don't match the expected pattern, invert them
    sv_profile['keep'] = False
    for index, row in sv_profile.iterrows():
        if row['svclass'] == "TRA" or (row['strand1'] == row['expected_strand1'] and row['strand2'] == row['expected_strand2']):
            sv_profile.at[index, 'keep'] = True
        elif row['strand1'] == row['expected_strand2'] and row['strand2'] == row['expected_strand1']:
            # Swap values
            sv_profile.at[index, 'start1'], sv_profile.at[index, 'start2'] = row['start2'], row['start1']
            sv_profile.at[index, 'chrom1'], sv_profile.at[index, 'chrom2'] = row['chrom2'], row['chrom1']
            sv_profile.at[index, 'strand1'], sv_profile.at[index, 'strand2'] = row['strand2'], row['strand1']
            sv_profile.at[index, 'keep'] = True
        else:
            sv_profile.at[index, 'keep'] = False
    
    # Generate end1 and end2
    sv_profile['end1'] = sv_profile['start1'] + 1
    sv_profile['end2'] = sv_profile['start2'] + 1

    # Remove SV with the wrong pattern
    sv_profile = sv_profile[sv_profile['keep']].drop(columns=['keep'])
    sv_profile.reset_index(drop=True, inplace=True)

    # Sort columns
    sv_profile = sv_profile[["chrom1", "start1", "end1", "chrom2", "start2", "end2", "strand1", "strand2", "svclass"]]

    return(sv_profile)

def check_inv_overlaps(sv_profile) -> pd.DataFrame:

    """
    Check if there is any overlap between inversions
    """

    # Check for overlaps within each chromosome
    sv_profile['n_overlaps'] = 0
    for chrom, group in sv_profile.groupby('chrom1'):
        group_indices:list = group.index
        for i in range(len(group_indices)):
            current_row:pd.Series = group.iloc[i]
            ## Find the next row
            n:int = 0
            while True:
                try:
                    next_row:pd.Series = group.iloc[i+1+n]
                    ## Check if there is an overlap
                    if current_row['start2'] > next_row['start1']:
                        n += 1
                    else:
                        sv_profile.loc[group_indices[i], 'n_overlaps'] = n
                        break
                except IndexError:
                    sv_profile.loc[group_indices[i], 'n_overlaps'] = n
                    break

    # Remove only events that overlap with events that also overlap
    sv_profile['keep'] = True
    for chrom, group in sv_profile.groupby('chrom1'):
        for idx, row in group.iterrows():
            n_overlaps:int = row['n_overlaps']
            if n_overlaps == 0:
                continue
            else:
                check_idx:list = list(range(idx + 1, idx + 1 + n_overlaps))
                overlap_sum:int = group.loc[group.index.isin(check_idx), 'n_overlaps'].sum()
                if overlap_sum != 0:
                    sv_profile.loc[idx, 'keep'] = False
                else:
                    continue
    
    sv_profile = sv_profile[sv_profile['keep']]
    sv_profile = sv_profile.drop(columns=['n_overlaps', 'keep']).reset_index(drop=True)

    return(sv_profile)

def check_tra_overlaps(sv_profile) -> pd.DataFrame:

    """
    Check if there is any overlap between translocations
    """

    # Pivot longer second chrom events
    sv_profile['sv_id'] = ['sv{}'.format(i) for i in range(len(sv_profile))]
    sv_profile_long = pd.concat([
        sv_profile[['sv_id', 'chrom1', 'start1', 'end1', 'strand1', 'svclass', 'id', 'allele']].rename(columns={'chrom1': 'chrom', 'start1': 'start', 'end1': 'end', 'strand1': 'strand'}),
        sv_profile[['sv_id', 'chrom2', 'start2', 'end2', 'strand2', 'svclass', 'id', 'allele']].rename(columns={'chrom2': 'chrom', 'start2': 'start', 'end2': 'end', 'strand2': 'strand'})])
    sv_profile_long.reset_index(drop=True, inplace=True)
    ## Sort the new dataframe
    sv_profile_long[["start", "end"]] = sv_profile_long[["start", "end"]].astype(int)
    sv_profile_long['chrom'] = sv_profile_long['chrom'].apply(chrom2int)
    sv_profile_long = sv_profile_long.sort_values(by=['chrom', 'start'], ignore_index=True)
    sv_profile_long['chrom'] = sv_profile_long['chrom'].apply(chrom2str)
    
    # Check for overlaps within each chromosome
    sv_profile_long['n_overlaps'] = 0
    for chrom, group in sv_profile_long.groupby('chrom'):
        group_indices:list = group.index
        for i in range(len(group_indices)):
            current_row:pd.Series = group.iloc[i]
            ## Find the next row
            n:int = 0
            while True:
                try:
                    next_row:pd.Series = group.iloc[i+1+n]
                    ## Check if there is an overlap
                    if current_row['end'] > next_row['start']:
                        n += 1
                    else:
                        sv_profile_long.loc[group_indices[i], 'n_overlaps'] = n
                        break
                except IndexError:
                    sv_profile_long.loc[group_indices[i], 'n_overlaps'] = n
                    break

    # Remove only events that overlap with events that also overlap
    sv_profile_long['keep'] = True
    for chrom, group in sv_profile_long.groupby('chrom'):
        for idx, row in group.iterrows():
            n_overlaps:int = row['n_overlaps']
            if n_overlaps == 0:
                continue
            else:
                check_idx:list = list(range(idx + 1, idx + 1 + n_overlaps))
                overlap_sum:int = group.loc[group.index.isin(check_idx), 'n_overlaps'].sum()
                if overlap_sum != 0:
                    sv_profile_long.loc[idx, 'keep'] = False
                else:
                    continue

    sv_profile_long = sv_profile_long[sv_profile_long['keep']]
    sv_profile_long = sv_profile_long.drop(columns=['n_overlaps', 'keep']).reset_index(drop=True)
    sv_profile_long = sv_profile_long.groupby('sv_id').filter(lambda group: len(group) == 2)

    # Pivot wider the dataframe to the original shape
    sv_profile = sv_profile_long.groupby('sv_id').apply(
        lambda group: pd.Series({
            'chrom1': group.iloc[0]['chrom'],
            'start1': group.iloc[0]['start'],
            'end1': group.iloc[0]['end'],
            'chrom2': group.iloc[1]['chrom'],
            'start2': group.iloc[1]['start'],
            'end2': group.iloc[1]['end'],
            'strand1': group.iloc[0]['strand'],
            'strand2': group.iloc[1]['strand'],
            'svclass': group.iloc[0]['svclass'],
            'id': group.iloc[0]['id'],
            'allele': group.iloc[0]['allele']})).reset_index()

    return(sv_profile)

def sort_sv(sv) -> pd.DataFrame: 

    """
    Sort SV dataframe
    """

    sv[["start1", "end1", "start2", "end2"]] = sv[["start1", "end1", "start2", "end2"]].astype(int)
    sv['chrom1'] = sv['chrom1'].apply(chrom2int)
    sv['chrom2'] = sv['chrom2'].apply(chrom2int)
    sv = sv.sort_values(by=['chrom1', 'start1', 'chrom2', 'start2'], ignore_index=True)
    sv['chrom1'] = sv['chrom1'].apply(chrom2str)
    sv['chrom2'] = sv['chrom2'].apply(chrom2str)
    
    return(sv)

def cna2sv_dupdel(cna) -> pd.DataFrame:

    """
    Automatically create DUP/DEL SVs based on CNA events
    """
    
    rows:list = []
    for _,row in cna.iterrows():
        chrom, start, end, major_cn, minor_cn, cna_id,  = row['chrom'], row['start'], row['end'], row['major_cn'], row['minor_cn'], row['id']
    
        # Duplications
        if major_cn > 1:
            rows.append({
                "chrom1": chrom, "start1": start, "end1": start + 1,
                "chrom2": chrom, "start2": end, "end2": end + 1,
                "strand1": "-", "strand2": "+", "svclass": "DUP",
                "id": cna_id, "allele": "major"})
        if minor_cn > 1:
            rows.append({
                "chrom1": chrom, "start1": start, "end1": start + 1,
                "chrom2": chrom, "start2": end, "end2": end + 1,
                "strand1": "-", "strand2": "+", "svclass": "DUP",
                "id": cna_id, "allele": "minor"})
        # Deletions
        if major_cn < 1:
            rows.append({
                "chrom1": chrom, "start1": start, "end1": start + 1,
                "chrom2": chrom, "start2": end, "end2": end + 1,
                "strand1": "+", "strand2": "-", "svclass": "DEL",
                "id": cna_id, "allele": "major"})
        if minor_cn < 1:
            rows.append({
                "chrom1": chrom, "start1": start, "end1": start + 1,
                "chrom2": chrom, "start2": end, "end2": end + 1,
                "strand1": "+", "strand2": "-", "svclass": "DEL",
                "id": cna_id, "allele": "minor"})
    
    sv_dupdel:pd.DataFrame = pd.DataFrame(rows)
    return(sv_dupdel)

def find_closest_range(row, cna) -> tuple:

    """
    Find the closest CNA event to each of the simulated SVs
    """

    cna_copy = cna.copy()

    # Remove homozygous CNA deletions events
    cna_hom_del:list = list(cna_copy.loc[cna_copy['major_cn'] == 0, 'id'])

    # Convert positions to a continous range
    keys:list = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']
    values:list = [0,249250621,492449994,690472424,881626700,1062541960,1233657027,1392795690,1539159712,1680373143,1815907890,1950914406,2084766301,2199936179,2307285719,2409817111,2500171864,2581367074,2659444322,2718573305,2781598825,2829728720,2881033286,3036303846]
    chrom_cumsum_dict:dict = dict(zip(keys, values))
    cna_copy['start_continous'] = cna_copy.apply(lambda x: x['start'] + chrom_cumsum_dict[str(x['chrom'])], axis=1)
    row['start1'] = int(row['start1']) + chrom_cumsum_dict[str(row['chrom1'])]
    row['start2'] = int(row['start2']) + chrom_cumsum_dict[str(row['chrom2'])]

    # Find closest start
    cna_copy['start_distance'] = cna_copy.apply(lambda x: (row['start1'] - x['start_continous']) if (row['start1'] - x['start_continous']) > 0 else float('inf'), axis=1)
    closest_start:pd.DataFrame = cna_copy.loc[cna_copy['start_distance'].idxmin()]
    closest_start.rename({"id": "start1_id"}, inplace=True)
    closest_start['start1_id'] = '-' if closest_start['start1_id'] in cna_hom_del else closest_start['start1_id']
    
    # Find closest end
    cna_copy['end_distance'] = cna_copy.apply(lambda x: (row['start2'] - x['start_continous']) if (row['start2'] - x['start_continous']) > 0 else float('inf'), axis=1)
    closest_end:pd.DataFrame = cna_copy.loc[cna_copy['end_distance'].idxmin()]
    closest_end.rename({"id": "start2_id"}, inplace=True)
    closest_end['start2_id'] = '-' if closest_end['start2_id'] in cna_hom_del else closest_end['start2_id']

    return(closest_start['start1_id'], closest_end['start2_id'])

def assign_inv_alleles(row, cna) -> str:

    """
    Assign major/minor allele tags for inversions
    """

    # Extract CNA information
    row['keep'] = True
    row['allele'] = np.random.choice(['major', 'minor'])
    if row['start1_id'] == row['start2_id']:
        row['id'] = row['start1_id']
        cn:pd.Series = cna.loc[cna['id'] == row['start1_id'], ['major_cn', 'minor_cn']].iloc[0]
    elif row['svclass'] == "h2hINV":
        row['id'] = row['start1_id']
        cn:pd.Series = cna.loc[cna['id'] == row['start1_id'], ['major_cn', 'minor_cn']].iloc[0]
    elif row['svclass'] == "t2tINV":
        row['id'] = row['start2_id']
        cn:pd.Series = cna.loc[cna['id'] == row['start2_id'], ['major_cn', 'minor_cn']].iloc[0]
    else:
        pass

    # Assign alleles
    if cn['major_cn'] == 0:
            row['keep'] = False
    elif cn['minor_cn'] == 0:
        row['allele'] = 'major'
    else:
        pass

    return(row['id'], row['allele'], row['keep'])

def assign_inv(cna, sv, sv_deldup) -> pd.DataFrame:

    """
    Assign simulated inversions to CNA events
    """

    # Assign the inversions based on CNA events
    sv_inv:pd.DataFrame = sv[sv['svclass'].isin(['h2hINV', 't2tINV'])]
    sv_inv = sv_inv.reset_index(drop=True)
    sv_inv = check_inv_overlaps(sv_inv)
    sv_inv[['start1_id', 'start2_id']] = sv_inv.apply(lambda row: find_closest_range(row, cna), axis=1, result_type='expand')
    sv_inv = sv_inv[(sv_inv['start1_id'] != '-') & (sv_inv['start2_id'] != '-')].reset_index(drop=True)
    sv_inv[['id', 'allele', 'keep']] = sv_inv.apply(lambda row: assign_inv_alleles(row, cna), axis=1, result_type='expand')
    sv_inv = sv_inv[sv_inv['keep']].reset_index(drop=True)
    sv_inv = sv_inv.drop(columns=['start1_id', 'start2_id', 'keep'])
    
    # Concatenate del, dup and inv
    sv_deldup_inv:pd.DataFrame = pd.concat([sv_deldup, sv_inv], ignore_index = True)
    sv_deldup_inv = sort_sv(sv_deldup_inv)

    return(sv_deldup_inv)

def assign_tra_alleles_len(row, cna) -> str:

    """
    Assign major/minor allele tags for translocations and define the length
    """

    # Extract CNA information
    cna = cna[cna['chrom'].isin([row['chrom1'], row['chrom2']])]
    ## In 20% of TRA create a more complex event
    if np.random.rand() < 0.2:
        cn_alt_id:str = np.random.choice(['start1_id', 'start2_id'])
        cn_norm_id:str = 'start1_id' if cn_alt_id == 'start2_id' else 'start2_id'

        cna_alt_id_value:str = f'cna{int(row[cn_alt_id].replace("cna", "")) + 1}'
        cna_norm_id_value:str = row[cn_norm_id]
        if cna_alt_id_value in cna['id'] and cna_norm_id_value in cna['id']:
            if cn_alt_id == 'start1_id':
                cn1:pd.Series = cna.loc[cna['id'] == cna_alt_id_value].squeeze()
            else:
                cn2:pd.Series = cna.loc[cna['id'] == cna_alt_id_value].squeeze()

            if cn_norm_id == 'start1_id':
                cn1:pd.Series = cna.loc[cna['id'] == cna_norm_id_value].squeeze()
            else:
                cn2:pd.Series = cna.loc[cna['id'] == cna_norm_id_value].squeeze()
        else:
            # In case the next CNA event is not located in the same chromosome
            cn1:pd.Series = cna.loc[cna['id'] == row['start1_id']].squeeze()
            cn2:pd.Series = cna.loc[cna['id'] == row['start2_id']].squeeze()
    else:
        cn1:pd.Series = cna.loc[cna['id'] == row['start1_id']].squeeze()
        cn2:pd.Series = cna.loc[cna['id'] == row['start2_id']].squeeze()
    
    # Select an allele
    row['allele1'] = np.random.choice(['major', 'minor'])
    row['allele2'] = np.random.choice(['major', 'minor'])
    if cn1['minor_cn'] == 0 and cn2['minor_cn'] == 0 and row['allele1'] == "minor" and row['allele2'] == "minor":
        change_allele:str = np.random.choice(['allele1', 'allele2'])
        row[change_allele] = 'major'

    # Define TRA length
    ## First chrom
    if row['strand1'] == '+':
        row['start1'] = cn1['start']
    else:
        row['end1'] = cn1['end']
    ## Second chrom
    if row['strand2'] == '+':
        row['start2'] = cn2['start']
    else:
        row['end2'] = cn2['end']
    
    # Adapt row shape
    row['id'] = f"{row['start1_id']},{row['start2_id']}"
    row['allele'] = f"{row['allele1']},{row['allele2']}"
    row.drop(labels=['start1_id', 'start2_id', 'allele1', 'allele2'], inplace=True)
    
    return(row)

def assign_tra(cna, sv, sv_deldup_inv) -> pd.DataFrame:

    """
    Assign simulated translocations to CNA events
    """

    # Assign the translocations based on CNA events
    sv_tra:pd.DataFrame = sv[sv['svclass']=='TRA']
    sv_tra = sv_tra.reset_index(drop=True)
    sv_tra[['start1_id', 'start2_id']] = sv_tra.apply(lambda row: find_closest_range(row, cna), axis=1, result_type='expand')
    sv_tra = sv_tra[(sv_tra['start1_id'] != '-') & (sv_tra['start2_id'] != '-')].reset_index(drop=True)
    sv_tra = sv_tra.apply(lambda row: assign_tra_alleles_len(row, cna), axis=1)
    sv_tra = check_tra_overlaps(sv_tra)

    # Concatenate all SV
    sv_deldup_inv_tra:pd.DataFrame = pd.concat([sv_deldup_inv, sv_tra], ignore_index = True)
    sv_deldup_inv_tra = sort_sv(sv_deldup_inv_tra)

    return(sv_deldup_inv_tra)

def align_cna_sv(cna, sv) -> pd.DataFrame:

    """
    Assign SV to CNA events
    """

    # Automatically create DUP/DEL SVs based on CNA events
    sv_assigned:pd.DataFrame = cna2sv_dupdel(cna)

    # Assign simulated INV based on CNA events
    if not sv[sv['svclass'].isin(['h2hINV', 't2tINV'])].empty:
        sv_assigned = assign_inv(cna, sv, sv_assigned)

    # Assign simulated TRA based on CNA events
    if not sv[sv['svclass']=='TRA'].empty:
        sv_assigned = assign_tra(cna, sv, sv_assigned)

    return(sv_assigned)

def simulate_sv(case_cna, nSV, tumor, svModel, gender, idx) -> pd.DataFrame:
    
    """
    Generate SVs
    """

    # Simulate the events
    case_sv:pd.DataFrame = pd.DataFrame()
    for sv_class in nSV.index.tolist():
        n:int = nSV[sv_class]
        if n != 0:
            concat_tmp_sv:pd.DataFrame = pd.DataFrame()
            while concat_tmp_sv.shape[0] < n:
                ## Simulate and assign SV positions
                tmp_sv:pd.DataFrame = svModel['sv'].generate_samples(n*5, var_column='svclass', var_class=sv_class)
                tmp_sv['start'] = get_sv_coordinates(n*5, svModel, gender)
                tmp_sv['len'] = round(np.exp(tmp_sv['len'])*10000)
                tmp_sv['end'] = tmp_sv['start'] + tmp_sv['len'] 
                tmp_sv = assign_chromosome(tmp_sv, sv=True, gender=gender)
                tmp_sv = check_sv_strand_patterns(tmp_sv)
                concat_tmp_sv = pd.concat([concat_tmp_sv, tmp_sv], ignore_index=True)
            
            concat_tmp_sv = concat_tmp_sv.sample(n = n, ignore_index=True)
            case_sv = pd.concat([case_sv, concat_tmp_sv])
        else:
            continue
    
    # Sort
    case_sv = sort_sv(case_sv)

    # Assign SV to CNA events
    case_sv = align_cna_sv(case_cna, case_sv)
    
    # Add donor and tumor columns
    case_sv["donor_id"] = f"sim{idx}"
    case_sv["tumor"] = tumor
    try:
        case_sv = case_sv.drop(columns=['sv_id'])
    except KeyError:
        pass

    return(case_sv)

@click.group()
def cli():
    pass

@click.command(name="availTumors")
def availTumors():
    
    """
    List of available tumors to simulate
    """

    tumors:list = [["Breast-AdenoCa", "CNS-PiloAstro", "Eso-AdenoCa", "Kidney-RCC", "Liver-HCC", "Lymph-CLL", "Panc-Endocrine", "Prost-AdenoCA"]]
    tumors:str = '\n'.join(['\t\t'.join(x) for x in tumors])
    click.echo(f"\nThis is the list of available tumor types that can be simulated using oncoGAN:\n\n{tumors}\n")

@click.command(name="vcfGANerator")
@click.option("-@", "--cpus",
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of CPUs to use")
@click.option("--tumor",
              type=click.Choice(["Breast-AdenoCa", "CNS-PiloAstro", "Eso-AdenoCa", "Kidney-RCC", "Liver-HCC", "Lymph-CLL", "Panc-Endocrine", "Prost-AdenoCA"]),
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
              required=True,
              help="hg19 reference genome in fasta format")
@click.option("--prefix",
              type=click.STRING,
              help="Prefix to name the output. If not, '--tumor' option is used as prefix")
@click.option("--outDir", "outDir",
              type=click.Path(exists=False, file_okay=False),
              default=os.getcwd(),
              show_default=False,
              help="Directory where save the simulations. Default is the current directory")
@click.option("--hg38", "hg38",
              is_flag=True,
              required=False,
              help="Transform the mutations to hg38")
@click.option("--mut/--no-mut", "simulateMuts",
              is_flag=True,
              required=False,
              default=True,
              show_default=True,
              help="Simulate mutations")
@click.option("--CNA-SV/--no-CNA-SV", "simulateCNA_SV",
              is_flag=True,
              required=False,
              default=True,
              show_default=True,
              help="Simulate CNA and SV events")
@click.option("--plots/--no-plots", "savePlots",
              is_flag=True,
              required=False,
              default=True,
              show_default=True,
              help="Save plots")
@click.version_option(version=VERSION,
                      package_name="OncoGAN",
                      prog_name="OncoGAN")
def oncoGAN(cpus, tumor, nCases, refGenome, prefix, outDir, hg38, simulateMuts, simulateCNA_SV, savePlots):

    """
    Command to simulate mutations (VCF) for different tumor types using a GAN model
    """
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    # Torch options
    device:str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load models
    countModel, mutModel, posModel, driversModel, countCorr, countEx = tumor_models(tumor, device)
    cna_sv_countModel, cnaModel, svModel = cna_sv_models(device)

    # Load reference genome
    fasta = Fasta(refGenome)

    # Generate the counts for each type of mutation for each case
    counts:pd.DataFrame = simulate_counts(tumor, countModel, nCases, countCorr, countEx)

    # Annotate VAF rank to each donor
    donors_vafRank:list = simulate_vaf_rank(tumor, nCases)

    # Generate CNA and SV counts for each case
    cna_sv_counts:pd.DataFrame = select_cna_sv_counts(cna_sv_countModel, nCases, tumor, counts)

    # Simulate one donor at a time
    muts:pd.DataFrame = pd.DataFrame()
    for idx in tqdm(range(nCases), desc = "Donors"):
        output:str = out_path(outDir, prefix, tumor, idx+1)
        
        # Focus in one case
        case_counts:pd.Series = counts.iloc[idx]
        nMut:int = int(case_counts.sum())
        case_rank:str = donors_vafRank[idx]
        case_cna_sv:pd.Series = cna_sv_counts.iloc[idx]

        # Detect specific tumor type in case we are simulating Lymph-CLL
        if tumor == 'Lymph-CLL':
            drivers_tumor:str = 'Lymph-MCLL' if case_counts['SBS9'] > 0 else 'Lymph-UCLL'
        else:
            drivers_tumor:str = tumor

        # Gender selection
        gender:str = gender_selection(tumor)

        if simulateMuts:
            # Simulate driver mutations to each donor
            case_drivers:pd.Series = simulate_drivers(drivers_tumor, driversModel)

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
            mut_vafs:list = simulate_mut_vafs(tumor, case_rank, nMut)
            case_muts['vaf'] = mut_vafs
            drivers_vafs:list = simulate_mut_vafs(tumor, case_rank, case_drivers.sum())

            # Create the VCF output
            vcf:pd.DataFrame = pd2vcf(case_muts, case_drivers, driversModel, drivers_vafs, drivers_tumor, fasta, idx)

            # Write the VCF
            ## Convert from hg19 to hg38
            if hg38:
                vcf = hg19tohg38(vcf=vcf)
            with open(output, "w+") as out:
                out.write("##fileformat=VCFv4.2\n")
                out.write(f"##fileDate={date.today().strftime('%Y%m%d')}\n")
                out.write(f"##source=OncoGAN-v{VERSION}\n")
                out.write(f"##reference={'hg38' if hg38 else 'hg19'}\n")
                out.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">\n')
                out.write('##INFO=<ID=MS,Number=A,Type=String,Description="Mutation type or mutational signature assigned to each mutation. Available options are: SBS (single base substitution signature), DNP (dinucleotide polymorphism), TNP (trinucleotide polymorphism), DEL (deletion), INS (insertion), driver* (driver mutation sampled from real donors)">\n')
            vcf.to_csv(output, sep="\t", index=False, mode="a")

        if simulateCNA_SV:
            # Simulate CNAs
            case_cna:pd.DataFrame = simulate_cnas(case_cna_sv['cna'], case_cna_sv['len'], tumor, cnaModel, gender, idx+1)

            # Simulate SVs
            case_sv:pd.DataFrame = simulate_sv(case_cna, case_cna_sv.loc['DEL':'t2tINV'], tumor, svModel, gender, idx+1)
            
            # Plots
            if savePlots:
                plot_cnas(case_cna, case_sv, tumor, output.replace(".vcf", "_cna.png"), idx+1) 
            
            # Convert from hg19 to hg38
            if hg38:
                case_cna = hg19tohg38(cna=case_cna)
                case_sv = hg19tohg38(sv=case_sv)

            # Save simulations
            case_cna.to_csv(output.replace(".vcf", "_cna.tsv"), sep ='\t', index=False)
            case_sv.to_csv(output.replace(".vcf", "_sv.tsv"), sep ='\t', index=False)

cli.add_command(availTumors)
cli.add_command(oncoGAN)
if __name__ == '__main__':
    cli()