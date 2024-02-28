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

    if tumor == "Lymph-CLL":
        # Counts model
        countModel:dict = {}
        countModel['MUT'] = torch.load(f"/genomeGAN/trained_models/counts/Lymph-MCLL_counts.pkl", map_location=device)
        countModel['UNMUT'] = torch.load(f"/genomeGAN/trained_models/counts/Lymph-UCLL_counts.pkl", map_location=device)

        # Mutations model
        mutModel:dict = {}
        mutModel['MUT'] = torch.load(f"/genomeGAN/trained_models/mutations/Lymph-CLL_mutations.pkl", map_location=device)
        mutModel['UNMUT'] = torch.load(f"/genomeGAN/trained_models/mutations/Lymph-CLL_mutations.pkl", map_location=device)
        
        # Drivers model and files
        driversModel:dict = {}
        driversModel['MUT']:dict = {}
        driversModel['MUT']['model'] = torch.load(f"/genomeGAN/trained_models/drivers/Lymph-MCLL_drivers.pkl", map_location=device)
        driversModel['MUT']['mutations'] = pd.read_csv(f"/genomeGAN/trained_models/drivers/Lymph-MCLL_driver_mutations.csv")
        driversModel['UNMUT']:dict = {}
        driversModel['UNMUT']['model'] = torch.load(f"/genomeGAN/trained_models/drivers/Lymph-UCLL_drivers.pkl", map_location=device)
        driversModel['UNMUT']['mutations'] = pd.read_csv(f"/genomeGAN/trained_models/drivers/Lymph-UCLL_driver_mutations.csv")

        # Positions model
        posModel:dict = {}
        with open(f"/genomeGAN/trained_models/positions/Lymph-MCLL_positions.pkl", 'rb') as f:
           posModel['MUT'] = pickle.load(f)
        with open(f"/genomeGAN/trained_models/positions/Lymph-UCLL_positions.pkl", 'rb') as f:
           posModel['UNMUT'] = pickle.load(f)
    else:
        # Counts model
        countModel = torch.load(f"/genomeGAN/trained_models/counts/{tumor}_counts.pkl", map_location=device)

        # Mutations model
        mutModel = torch.load(f"/genomeGAN/trained_models/mutations/{tumor}_mutations.pkl", map_location=device)
        
        # Drivers model and files
        driversModel:dict = {}
        driversModel['model'] = torch.load(f"/genomeGAN/trained_models/drivers/{tumor}_drivers.pkl", map_location=device)
        driversModel['mutations'] = pd.read_csv(f"/genomeGAN/trained_models/drivers/{tumor}_driver_mutations.csv")

        # Positions model
        with open(f"/genomeGAN/trained_models/positions/{tumor}_positions.pkl", 'rb') as f:
           posModel = pickle.load(f)
    
    return(countModel, mutModel, posModel, driversModel)

def out_path(outDir, prefix, tumor, n) -> click.Path:

    """
    Get the absolute path and name for the outputs
    """

    if prefix is not None:
        output:click.Path = f"{outDir}/{prefix}_sim{n}.vcf"
    else:
        output:click.Path = f"{outDir}/{tumor}_sim{n}.vcf"
    
    return(output)

def remove_low_expressed(row) -> list:
    
    """
    Quick function to process the counts and remove the mutations from very low expressed signatures
    """
    
    threshold = row.sum() * 0.05
    if threshold > 150: #Just in case the tumour has a lot of mutations 
        threshold = 150
    removed_values = row[row < threshold].sum()
    row = row.where(row >= threshold, 0)
    return(row,removed_values)

def preprocess_counts(counts) -> pd.DataFrame:
    
    """
    Function to preprocess the counts
    """
    
    # Restart the index
    counts.reset_index(drop=True, inplace=True)

    # Get signature columns
    sbs_columns:list = [col for col in counts.columns if col.startswith("SBS")]
    
    # Remove mutations from very low expressed signatures
    apply_res = counts[sbs_columns].apply(remove_low_expressed, axis=1, result_type='reduce')
    counts[sbs_columns] = apply_res.apply(lambda x:x[0])
    removed_sum = apply_res.apply(lambda x:x[1])

    # Distribute the removed mutations across the other signatures
    for idx, row in counts.iterrows():
        total_removed:int = removed_sum[idx]
        sbs_to_update:list = [col for col in sbs_columns if row[col] >= 15]

        if sbs_to_update:
            distribute_per_column:int = total_removed // len(sbs_to_update)
            remainder:int = total_removed % len(sbs_to_update)
            
            for col in sbs_to_update:
                to_add:int = distribute_per_column
                if remainder > 0:
                    to_add += 1
                    remainder -= 1
                counts.at[idx, col] += to_add
    
    # Check that there is a column for each mutation and if not initialize it
    counts = counts.assign(**{col:0 for col in ["DNP", "TNP", "INS", "DEL"] if col not in counts.columns})
    counts = counts.round(0).astype(int)
    
    return(counts)

def simulate_counts(tumor, countSynthesizer, nCases) -> pd.DataFrame:

    """
    Function to generate the number of each type of mutation per case
    """

    if tumor == "Lymph-CLL":
        mCases:int = round(nCases*0.42)
        uCases:int = nCases - mCases

        # MCLL
        m_counts:pd.DataFrame = pd.DataFrame()
        for _ in range(4):
            tmp:pd.DataFrame = countSynthesizer['MUT'].generate_samples(mCases)
            m_counts = pd.concat([m_counts,tmp])
        m_counts = m_counts.sample(n=mCases)

        # UCLL
        u_counts:pd.DataFrame = pd.DataFrame()
        for _ in range(4):
            tmp:pd.DataFrame = countSynthesizer['UNMUT'].generate_samples(uCases)
            u_counts = pd.concat([u_counts,tmp])
        u_counts = u_counts.sample(n=uCases)

        # Merge
        counts:pd.DataFrame = pd.concat([m_counts, u_counts])
        counts.fillna(0, inplace=True)
        counts.reset_index(drop=True, inplace=True)
        counts = counts.round(0).astype(int)

    else:
        counts:pd.DataFrame = pd.DataFrame()
        for _ in range(4):
            tmp:pd.DataFrame = countSynthesizer.generate_samples(nCases)
            counts = pd.concat([counts,tmp])
        counts = counts.sample(n=nCases)
        counts.reset_index(drop=True, inplace=True)

    # Preprocess the counts
    counts = preprocess_counts(counts)

    return(counts)

def simulate_vaf_rank(tumor, nCases) -> list:

    """
    Function to simulate the VAF range for each donor
    """

    rank_file:pd.DataFrame = pd.read_csv("/genomeGAN/trained_models/vaf_rank_list.tsv", sep='\t') 
    rank_file = rank_file.loc[rank_file["study"]==tumor]
    donor_vafs:list = random.choices(rank_file.columns[1:], weights=rank_file.values[0][1:], k=nCases)

    return(donor_vafs)

def simulate_drivers(tumor, driversSynthesizer, nCases) -> pd.DataFrame:

    """
    Function to simulate the driver mutations for each donor
    """

    if tumor == "Lymph-CLL":
        mCases:int = round(nCases*0.42)
        uCases:int = nCases - mCases
        
        # MCLL
        m_drivers:pd.DataFrame = driversSynthesizer['MUT']['model'].generate_samples(mCases)

        # UCLL
        u_drivers:pd.DataFrame = driversSynthesizer['UNMUT']['model'].generate_samples(uCases)

        # Merge
        drivers:pd.DataFrame = pd.concat([m_drivers, u_drivers])
        drivers.fillna(0, inplace=True)
        drivers.reset_index(drop=True, inplace=True)

    else:
        drivers:pd.DataFrame = driversSynthesizer['model'].generate_samples(nCases)
    
    drivers = drivers.round(0).astype(int)

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

def manullay_simulate_dnp_tnp(mutSynthesizer, nMut, mut_type) -> pd.DataFrame:
    
        """
        Function to manually simulate DNP and TNP mutations
        """
    
        nt_dict = {'A':['C', 'G', 'T'], 'C':['A', 'G', 'T'], 'G':['A', 'C', 'T'], 'T':['A', 'C', 'G']}
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

def simulate_mutations(mutSynthesizer, muts, nMut, case_counts) -> pd.DataFrame:

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
                dnp_muts = manullay_simulate_dnp_tnp(mutSynthesizer, oriVSsim_types['DNP'], 'DNP')
                muts = pd.concat([muts,dnp_muts])
                oriVSsim_types['DNP'] = 0
            if oriVSsim_types['TNP'] > 0:
                tnp_muts = manullay_simulate_dnp_tnp(mutSynthesizer, oriVSsim_types['TNP'], 'TNP')
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
        start:int = row['start']
        end:int = row['end']
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
            driver_rows:pd.DataFrame = drivers_mutations['MUT']['mutations'][drivers_mutations['driver'] == driver]
        elif drivers_tumor == 'Lymph-UCLL':
            driver_rows:pd.DataFrame = drivers_mutations['UNMUT']['mutations'][drivers_mutations['driver'] == driver]
        else:
            driver_rows:pd.DataFrame = drivers_mutations['mutations'][drivers_mutations['driver'] == driver]
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
    countModel, mutModel, posModel, driversModel = tumor_models(tumor, device)

    # Load reference genome
    fasta = Fasta(refGenome)

    # Generate the counts for each type of mutation for each case
    counts:pd.DataFrame = simulate_counts(tumor, countModel, nCases)

    # Annotate VAF rank to each donor
    donors_vafRank:list = simulate_vaf_rank(tumor, nCases)

    # Simulate driver mutations to each donor
    donors_drivers: pd.DataFrame = simulate_drivers(tumor, driversModel, nCases)

    # Simulate one donor at a time
    muts:pd.DataFrame = pd.DataFrame()
    for idx in tqdm(range(nCases), desc = "Donors"):
        # Focus in one case
        case_counts:pd.Series = counts.iloc[idx]
        nMut:int = int(case_counts.sum())
        case_rank:str = donors_vafRank[idx]
        case_drivers:pd.Series = donors_drivers.iloc[idx]

        # Detect specific tumor type in case we are simulating Lymph-CLL
        if tumor == 'Lymph-CLL':
            drivers_tumor:str = 'Lymph-MCLL' if case_counts['SBS9'] > 0 else 'Lymph-UCLL'
        else:
            drivers_tumor:str = tumor

        # Gender selection
        gender:str = gender_selection(tumor)

        # Generate the mutations
        muts, case_muts = simulate_mutations(mutModel, muts, nMut, case_counts)
        
        # Select the mutations corresponding for this case
        muts, case_muts = select_case_mutations(muts, case_counts)

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