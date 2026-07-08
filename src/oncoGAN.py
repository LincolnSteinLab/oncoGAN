#!/usr/local/bin/python3

import sys
sys.path.append('/oncoGAN/models')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import re
import tempfile
from time import time
import click
import pickle
import json
import itertools
import subprocess
from multiprocessing import Pool, set_start_method
import random
import pandas as pd
import numpy as np
from typing import Literal, NamedTuple, Sequence
from datetime import date
from liftover import ChainFile
from tqdm import tqdm
from pyfaidx import Fasta
from Bio.Seq import Seq
import pyranges as pr
from cna_annotation import TumorGenome

VERSION = "1.0.0"

default_tumors:list[str] = ["Biliary-AdenoCA","Bladder-TCC","Bone-Leiomyo","Bone-Osteosarc","Breast-AdenoCa","Cervix-SCC","CNS-GBM","CNS-Medullo","CNS-Oligo","CNS-PiloAstro","ColoRect-AdenoCA","Eso-AdenoCa","Head-SCC","Kidney-ChRCC","Kidney-RCC","Liver-HCC","Lung-AdenoCA","Lung-SCC","Lymph-BNHL","Lymph-CLL","Myeloid-MPN","Ovary-AdenoCA","Panc-AdenoCA","Panc-Endocrine","Prost-AdenoCA","Skin-Melanoma","Stomach-AdenoCA","Thy-AdenoCA","Uterus-AdenoCA"]
default_cna_tumors:dict[str, list[str]] = {
    "Biliary-AdenoCA": ["main"], 
    "Bladder-TCC": ["BLCA"],
    "Bone-Leiomyo": ["main"],
    "Bone-Osteosarc": ["main"],
    "Breast-AdenoCa": ["BRCA"],
    "Cervix-SCC": ["CESC"],
    "CNS-GBM": ["GBM"],
    "CNS-Medullo": ["main"],
    "CNS-Oligo": ["LGG"],
    "CNS-PiloAstro": ["main"],
    "ColoRect-AdenoCA": ["COAD", "READ"],
    "Eso-AdenoCa": ["ESCA"],
    "Head-SCC": ["HNSC"],
    "Kidney-ChRCC": ["main"],
    "Kidney-RCC": ["KIRC"],
    "Liver-HCC": ["LIHC"],
    "Lung-AdenoCA": ["LUAD"],
    "Lung-SCC": ["LUSC"],
    "Lymph-BNHL": ["main"],
    "Lymph-CLL": ["main"],
    "Myeloid-MPN": ["main"],
    "Ovary-AdenoCA": ["OV"],
    "Panc-AdenoCA": ["PAAD"],
    "Panc-Endocrine": ["main"],
    "Prost-AdenoCA": ["PRAD"],
    "Skin-Melanoma": ["SKCM"],
    "Stomach-AdenoCA": ["STAD"],
    "Thy-AdenoCA": ["main"],
    "Uterus-AdenoCA": ["UCEC"]
}

#################
# Miscellaneous #
#################

def validate_template(template:str, default_tumors:list[str]) -> pd.DataFrame:

    """
    Check that the template file provided by the user is correct
    """

    df:pd.DataFrame = pd.read_csv(template)

    # Check "tumor" values exist in default_tumors
    invalid_tumors:set = set(df["Tumor"]) - set(default_tumors)
    if invalid_tumors:
        raise ValueError(f"Invalid tumor values found: {sorted(invalid_tumors)}. Run 'availTumors' subcommand to check the list of available tumors.")
    
    # Check "NinT" is float between [0, 1]
    df["NinT"] = pd.to_numeric(df["NinT"], errors="coerce").fillna(0)
    invalid_nint:pd.Series = (df["NinT"] < 0) | (df["NinT"] > 1)
    if invalid_nint.any():
        bad_rows:pd.DataFrame = df.loc[invalid_nint, ["NinT"]]
        raise ValueError(f"'NinT' must be a float between 0 and 1. Invalid values found:\n{bad_rows}")

    # Convert signature values to integers
    cols_to_int:pd.Index = df.columns.difference(["ID", "Tumor", "NinT"])
    df[cols_to_int] = (df[cols_to_int].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int))

    return df

def chrom2int(chrom:str) -> int:

    """
    Convert the chromosome to an integer
    """

    if chrom == 'X':
        return 23
    elif chrom == 'Y':
        return 24
    else:
        return int(chrom)

def chrom2str(chrom:int) -> str:

    """
    Convert the chromosome to a string
    """

    if chrom == 23:
        return 'X'
    elif chrom == 24:
        return 'Y'
    else:
        return str(chrom)

def hg19tohg38(vcf:pd.DataFrame) -> pd.DataFrame:

    """
    Convert hg19 coordinates to hg38
    """

    vcf_f = vcf.copy()
    converter:ChainFile = ChainFile('/.liftover/hg19ToHg38.over.chain.gz')
    for row in vcf_f.itertuples():
        chrom:str = str(row[1])
        pos:int = int(row[2])
        try:
            liftOver_result:tuple[str,int,str] = converter[chrom][pos][0]
            vcf_f.at[row.Index, '#CHROM'] = liftOver_result[0]
            vcf_f.at[row.Index, 'POS'] = liftOver_result[1]
        except IndexError:
            vcf_f.at[row.Index, '#CHROM'] = 'Remove'

    vcf_f = vcf_f[~vcf_f['#CHROM'].str.contains('Remove', na=False)]
    return(vcf_f)

##########
# Models #
##########

def calo_forest_generation(load_dir:str, y_labels:list[str]|tuple[str,...]) -> pd.DataFrame:
    
    """
    Generate samples using Calo-Forest models
    """
    
    # Load the forest model
    model:str = os.path.join(load_dir, 'forest_model.pkl')
    with open(model, 'rb') as file:
        model_dict:dict = pickle.load(file)
    model_dict['model'].set_logdir(load_dir)
    model_dict['model'].set_solver_fn(model_dict['cfg']["solver"])
    reverse_mapping:dict = {v: k for k, v in model_dict['mapping'].items()}

    # Prepare the number and type of samples to generate
    y_labels_map:list[int] = [reverse_mapping[x] for x in y_labels]
    n:int = len(y_labels_map)
    y:np.ndarray = np.array(y_labels_map)

    # Generate the samples
    Xy_fake:np.ndarray = model_dict['model'].generate(batch_size=n, label_y=y)

    # Map back the labels to their original values
    Xy_fake_df:pd.DataFrame = pd.DataFrame.from_records(Xy_fake, columns=model_dict['columns'])
    pred_col:str = Xy_fake_df.columns[-1]
    Xy_fake_df[pred_col] = Xy_fake_df[pred_col].apply(lambda x: model_dict['mapping'][x])

    del model #release model memory
    return Xy_fake_df

def dae_reconstruction(z:pd.DataFrame, dae_model:Literal['genomic_profile']) -> pd.DataFrame:
    
    """
    Function to reconstruct diffusion latent spaces using the DAE model
    """

    # Serialize the latent space
    z_serialized:bytes = pickle.dumps(z)

    # Run the reconstruction in another environment
    command:Sequence[str] = ['micromamba', 'run', '-n', 'dae',
                    'python3', '/oncoGAN/dae_reconstruction.py',
                    '--model', f'{dae_model}']
    reconstructed_serialized = subprocess.run(command,
                                              input=z_serialized,
                                              capture_output=True)
    
    # Load the results
    reconstructed:dict = pickle.loads(reconstructed_serialized.stdout)
    reconstructed_df:pd.DataFrame = pd.DataFrame(reconstructed['df'], columns=reconstructed['columns'])

    return reconstructed_df

###############
# Simulations #
###############

def simulate_counts(tumor_f:str, nCases_f:int) -> pd.DataFrame:

    """
    Function to generate the number of each type of mutation per case
    """

    def clean_counts_apply(row:pd.Series) -> pd.Series:

        """
        Apply function to adapt the raw counts from the generator
        """

        tumor:str = row['Tumor']
        for col in row.index[:-1]:
            val:float = round(row[col])
            stats:dict = tumor_stats.get(tumor, {}).get(col)

            if stats is None:
                row[col] = val
            elif stats.get('All_Zero', False):
                row[col] = 0
            elif val < stats['Min'] * 0.8:
                row[col] = 0
            elif val > stats['Max'] * 1.3:
                row[col] = np.nan
            else:
                row[col] = val
        return row
    
    # Prepare the list of donors to simulate
    nCases_x5:int = nCases_f * 5
    if tumor_f == "Lymph-CLL":
        mCases:int = round(nCases_x5*0.42)
        uCases:int = nCases_x5 - mCases
        cases_list:list[str] = ['Lymph-MCLL']*mCases + ['Lymph-UCLL']*uCases
    else:
        cases_list:list[str] = [tumor_f]*nCases_x5
    
    # Generate samples
    counts:pd.DataFrame = calo_forest_generation('/oncoGAN/trained_models/donor_characteristics', cases_list)

    # Clean the output a bit (round, min and max boundaries)
    tumor_stats:dict = pd.read_pickle('/oncoGAN/trained_models/donor_characteristics/donor_characteristics_stats.pkl')
    counts = counts.apply(clean_counts_apply, axis=1).dropna().reset_index(drop=True)
    counts = counts.sample(n=nCases_f, replace=False).reset_index(drop=True)

    return counts

def simulate_sex(tumor_list_f:tuple[str, ...]) -> list[str]:

    """
    Simulate the sex of each donor
    """

    # Import the data
    tumor_list_f_updated:list[str] = ["Lymph-CLL" if tumor in {"Lymph-MCLL", "Lymph-UCLL"} else tumor for tumor in tumor_list_f]
    tumor_sex_df:pd.DataFrame = pd.read_csv('/oncoGAN/trained_models/xy_usage_ranks.txt', sep='\t')
    tumor_sex_list:list[str] = tumor_sex_df['label'].to_list()

    sex_list:list[str] = []
    for tumor in tumor_list_f_updated:
        tumor_sex_options:list[str] = [t[-1] for t in tumor_sex_list if t.startswith(tumor)]
        
        # Simulate sex
        if 'F' in tumor_sex_options and 'M' in tumor_sex_options:
            sex:str = "M" if random.random() < 0.5 else "F"
        elif 'F' in tumor_sex_options and 'M' not in tumor_sex_options:
            sex:str = "F"
        elif 'F' not in tumor_sex_options and 'M' in tumor_sex_options:
            sex:str = "M"
        else:
            continue
        
        sex_list.append(sex)
            
    return sex_list

def simulate_signatures(counts_f:pd.DataFrame) -> dict:

    """
    Function to simulate the mutational signatures for each donor
    """
    
    def assign_indel_ref_alt_apply(context:str) -> tuple[str,str]:

        """
        Apply function to assign the correct reference and alternative allele for indels
        """

        ref_list:list[str] = []
        alt_list:list[str] = []
        size, indel_type, base, context_length = context.split(':')
        size:int = int(size)
        context_length:int = int(context_length)
        if int(size) == 1:
            # Cosmic references are C and T nucleotides, but real mutations can also be the complementary nucleotides
            if base == 'C':
                repeat_base:str = random.choice(['C', 'G'])
            elif base == 'T':
                repeat_base:str = random.choice(['A', 'T'])

            if indel_type == 'Del':
                ref_list.append(repeat_base*(context_length+1))
                alt_list.append(repeat_base*context_length)
            elif indel_type == 'Ins':
                ref_list.append(repeat_base*context_length)
                alt_list.append(repeat_base*(context_length+1))
        elif base == "R":
            for _ in range(5):
                repeat_base:str = ''
                for _ in range(int(size)):
                    repeat_base +=  random.choice(['A', 'C', 'G', 'T'])

                if indel_type == 'Del':
                    ref_list.append(repeat_base*(context_length+1))
                    alt_list.append(repeat_base*context_length)
                elif indel_type == 'Ins':
                    ref_list.append(repeat_base*context_length)
                    alt_list.append(repeat_base*(context_length+1))
        elif base == 'M': 
            for _ in range(5):
                repeat_base:str = ''
                range_len:int = int(size) if int(size) < 5 else random.randint(5, 9)
                context_length_updated:int = context_length if context_length < 5 else random.randint(4, range_len-1)
                for _ in range(range_len):
                    if len(repeat_base) == 0:
                        repeat_base +=  random.choice(['A', 'C', 'G', 'T'])
                    else:
                        repeat_base +=  random.choice([nt for nt in ['A', 'C', 'G', 'T'] if nt != repeat_base[-1]])

                ref1:str = repeat_base+repeat_base[:context_length_updated]
                ref2:str = repeat_base[-context_length_updated:]+repeat_base
                ref_list.append(f"{ref1}|{ref2}")
                alt_list.append(f"NA|{repeat_base[-1]}")
        
        ref:str = ','.join(ref_list)
        alt:str = ','.join(alt_list)
        
        return (ref, alt)
    
    def reverse_complement_sbs_apply(row:pd.Series) -> tuple[str, str, str]:

        """
        Apply function to compute the reverse complement for reference and alternative alleles
        """
        
        if random.choice([True, False]):
            context:str = str(Seq(row['contexts']).reverse_complement())
            ref:str = str(Seq(row['ref']).reverse_complement())
            alt:str = str(Seq(row['alt']).reverse_complement())
            return(context, ref, alt)
        else:
            return(row['contexts'], row['ref'], row['alt'])

    def process_mutations(mut_df:pd.DataFrame, mut_type:Literal['sbs', 'id']) -> pd.DataFrame:

        """
        Function to process the mutations associated with signatures
        """

        def distribute_diff_apply(group:pd.DataFrame) -> pd.DataFrame:

            """
            Apply function to calculate the difference between the observed and the expected number of mutations
            """
            
            total:int = int(group['total'].iloc[0])
            current:int = group['n'].sum()
            missing:int = total - current

            if missing > 0:
                idx:pd.Index = group['diff'].sample(frac=1).nlargest(missing).index
                group.loc[idx, 'n'] += 1

            return group
        
        # Normalize the context usage
        contexts:pd.DataFrame = mut_df.drop(columns=['signature', 'total'])
        normalized_contexts:pd.DataFrame = contexts.div(contexts.sum(axis=1), axis=0)
        normalized_contexts = pd.concat([mut_df['signature'], normalized_contexts], axis=1)
        
        # Pivot longer
        mut_df_long:pd.DataFrame = pd.melt(normalized_contexts, id_vars=['signature'], var_name='contexts', value_name='perc')
        
        # Calculate the number of mutations for each context
        mut_df_long['perc'] = pd.to_numeric(mut_df_long['perc'], errors='coerce')
        mut_df_long['total'] = pd.to_numeric(mut_df_long['signature'].map(row_features), errors='coerce')
        mut_df_long['exp'] = mut_df_long['perc'] * mut_df_long['total']
        mut_df_long['n'] = np.floor(mut_df_long['exp']).astype(int)
        mut_df_long['diff'] = mut_df_long['exp'] - mut_df_long['n']
        mut_df_long = mut_df_long.groupby('signature', group_keys=False).apply(distribute_diff_apply)
        mut_df_long = mut_df_long.drop(columns=['exp', 'diff'])
        mut_df_long_expanded:pd.DataFrame = mut_df_long.loc[mut_df_long.index.repeat(mut_df_long['n'].astype(int))].reset_index(drop=True)

        if mut_type == 'sbs':
            # Extract context, ref and alt bases
            mut_df_long_expanded[['pre', 'ref', 'alt', 'post']] = mut_df_long_expanded['contexts'].str.extract(r'([A-Z])\[([A-Z])>([A-Z])\]([A-Z])')
            mut_df_long_expanded['contexts'] = mut_df_long_expanded['pre'] + mut_df_long_expanded['ref'] + mut_df_long_expanded['post']
            mut_df_long_expanded[['contexts', 'ref', 'alt']] = mut_df_long_expanded.apply(reverse_complement_sbs_apply, axis=1, result_type='expand')
            mut_df_processed:pd.DataFrame = mut_df_long_expanded[['signature', 'contexts', 'ref', 'alt']]
        elif mut_type == 'id':
            mut_df_long_expanded[['ref', 'alt']] = mut_df_long_expanded['contexts'].apply(assign_indel_ref_alt_apply).apply(pd.Series)
            mut_df_processed:pd.DataFrame = mut_df_long_expanded[['signature', 'contexts', 'ref', 'alt']]
            
        return mut_df_processed[['signature', 'contexts', 'ref', 'alt']]
    
    # Count how many donors present each signature
    signatures2sim:pd.Series = (counts_f[counts_f.columns[:-1]] != 0).sum()

    # Prepare the list of signatures to simulate
    sbs_list:list[str] = [feature for feature, count in signatures2sim.items() for _ in range(count*100) if feature.startswith("SBS")]
    indel_list:list[str] = [feature for feature, count in signatures2sim.items() for _ in range(count*100) if feature.startswith("ID")]

    # Generate the signatures
    sbs:pd.DataFrame|None
    if len(sbs_list) > 0:
        sbs = calo_forest_generation('/oncoGAN/trained_models/sbs_context', sbs_list)
    else:
        sbs = None
    indels:pd.DataFrame|None
    if len(indel_list) > 0:
        indels = calo_forest_generation('/oncoGAN/trained_models/indel_context', indel_list)
    else:
        indels = None
    
    # Assign signatures for each donor
    nucleotides:list[str] = ['A', 'C', 'G', 'T']
    signatures_dict:dict = {}
    for donor_id, row in counts_f.iterrows():
        row_features:pd.Series = row[row != 0]
        total:int = row_features.sum()

        ## Create the empty dataframes
        signatures_donor_sbs:pd.DataFrame = pd.DataFrame()
        signatures_donor_indels:pd.DataFrame = pd.DataFrame()
        donor_dnp:pd.DataFrame = pd.DataFrame()
        donor_tnp:pd.DataFrame = pd.DataFrame()
        donor_medium_ins:pd.DataFrame = pd.DataFrame()
        donor_big_ins:pd.DataFrame = pd.DataFrame()
        donor_medium_del:pd.DataFrame = pd.DataFrame()
        donor_big_del:pd.DataFrame = pd.DataFrame()
        for signature, count in row_features.items():
            ## Find the signature where the difference between 'total' and this donor's total number of mutations is minimized
            if signature.startswith("SBS") and sbs is not None:
                sbs_signature:pd.DataFrame = sbs[sbs['signature'] == signature]
                idx:int = (sbs_signature['total'] - total).abs().idxmin()
                selected_sbs:pd.DataFrame = sbs_signature.loc[idx].to_frame().T
                selected_sbs['total'] = count
                signatures_donor_sbs = pd.concat([signatures_donor_sbs, selected_sbs], ignore_index=True)
                sbs = sbs.drop(idx).reset_index(drop=True)
            elif signature.startswith("ID") and indels is not None:
                indels_signature:pd.DataFrame = indels[indels['signature'] == signature]
                idx:int = (indels_signature['total'] - total).abs().idxmin()
                selected_indels:pd.DataFrame = indels_signature.loc[idx].to_frame().T
                selected_indels['total'] = count
                signatures_donor_indels = pd.concat([signatures_donor_indels, selected_indels], ignore_index=True)
                indels = indels.drop(idx).reset_index(drop=True)
            elif signature == 'DNP':
                while donor_dnp.shape[0] != count:
                    dnp_ref:str = random.choice([f"{n1}{n2}" for n1, n2 in itertools.product(nucleotides, repeat=2)])
                    n1_alt_list:list[str] = [nt for nt in nucleotides if nt != dnp_ref[0]]
                    n2_alt_list:list[str] = [nt for nt in nucleotides if nt != dnp_ref[1]]
                    dnp_alt:str = random.choice([f"{n1}{n2}" for n1, n2 in itertools.product(n1_alt_list, n2_alt_list)])
                    selected_dnp:pd.DataFrame = pd.DataFrame({'signature':['DNP'], 'contexts':['DNP'], 'ref':[dnp_ref], 'alt':[dnp_alt]})
                    donor_dnp = pd.concat([donor_dnp, selected_dnp], ignore_index=True)
            elif signature == 'TNP':
                while donor_tnp.shape[0] != count:
                    tnp_ref:str = random.choice([f"{n1}{n2}{n3}" for n1, n2, n3 in itertools.product(nucleotides, repeat=3)])
                    n1_alt_list:list[str] = [nt for nt in nucleotides if nt != tnp_ref[0]]
                    n2_alt_list:list[str] = [nt for nt in nucleotides if nt != tnp_ref[1]]
                    n3_alt_list:list[str] = [nt for nt in nucleotides if nt != tnp_ref[2]]
                    tnp_alt:str = random.choice([f"{n1}{n2}{n3}" for n1, n2, n3 in itertools.product(n1_alt_list, n2_alt_list, n3_alt_list)])
                    selected_tnp:pd.DataFrame = pd.DataFrame({'signature':['TNP'], 'contexts':['TNP'], 'ref':[tnp_ref], 'alt':[tnp_alt]})
                    donor_tnp = pd.concat([donor_tnp, selected_tnp], ignore_index=True)
            elif signature == 'medium_ins':
                while donor_medium_ins.shape[0] != count:
                    medium_ins_ref:str = random.choice(nucleotides)
                    medium_ins_alt:str = ''.join(random.choices(nucleotides, k=random.randint(6,10)))
                    selected_medium_ins:pd.DataFrame = pd.DataFrame({'signature':['medium_ins'], 'contexts':['medium_ins'], 'ref':[medium_ins_ref], 'alt':[medium_ins_alt]})
                    donor_medium_ins = pd.concat([donor_medium_ins, selected_medium_ins], ignore_index=True)
            elif signature == 'big_ins':
                while donor_big_ins.shape[0] != count:
                    big_ins_ref:str = random.choice(nucleotides)
                    big_ins_alt:str = ''.join(random.choices(nucleotides, k=random.randint(11,25)))
                    selected_big_ins:pd.DataFrame = pd.DataFrame({'signature':['big_ins'], 'contexts':['big_ins'], 'ref':[big_ins_ref], 'alt':[big_ins_alt]})
                    donor_big_ins = pd.concat([donor_big_ins, selected_big_ins], ignore_index=True)
            elif signature == 'medium_del':
                while donor_medium_del.shape[0] != count:
                    medium_del_ref:int = random.randint(6,10)
                    medium_del_alt:str = random.choice(nucleotides)
                    selected_medium_del:pd.DataFrame = pd.DataFrame({'signature':['medium_del'], 'contexts':['medium_del'], 'ref':[medium_del_ref], 'alt':[medium_del_alt]})
                    donor_medium_del = pd.concat([donor_medium_del, selected_medium_del], ignore_index=True)
            elif signature == 'big_del':
                while donor_big_del.shape[0] != count:
                    big_del_ref:int = random.randint(11,25)
                    big_del_alt:str = random.choice(nucleotides)
                    selected_big_del:pd.DataFrame = pd.DataFrame({'signature':['big_del'], 'contexts':['big_del'], 'ref':[big_del_ref], 'alt':[big_del_alt]})
                    donor_big_del = pd.concat([donor_big_del, selected_big_del], ignore_index=True)
        
        if sbs is not None:
            signatures_donor_sbs = process_mutations(signatures_donor_sbs, mut_type='sbs')
        if indels is not None:
            signatures_donor_indels = process_mutations(signatures_donor_indels, mut_type='id')

        signatures_dict[donor_id] = pd.concat([signatures_donor_sbs, signatures_donor_indels, donor_dnp, donor_tnp, donor_medium_ins, donor_big_ins, donor_medium_del, donor_big_del], ignore_index=True)

    return signatures_dict

def simulate_genomic_profile(tumor_list_f:tuple[str, ...], counts_total_f:pd.Series, sex_f:list[str]) -> pd.DataFrame:

    """
    Function to simulate the genomic pattern profiles for each donor
    """
    
    Y_cols:list[str] = ["[3.036e+09,3.037e+09)","[3.038e+09,3.039e+09)","[3.039e+09,3.04e+09)","[3.04e+09,3.041e+09)","[3.041e+09,3.042e+09)","[3.042e+09,3.043e+09)","[3.043e+09,3.044e+09)","[3.044e+09,3.045e+09)","[3.045e+09,3.046e+09)","[3.046e+09,3.047e+09)","[3.049e+09,3.05e+09)","[3.05e+09,3.051e+09)","[3.051e+09,3.052e+09)","[3.052e+09,3.053e+09)","[3.053e+09,3.054e+09)","[3.054e+09,3.055e+09)","[3.055e+09,3.056e+09)","[3.056e+09,3.057e+09)","[3.057e+09,3.058e+09)","[3.058e+09,3.059e+09)","[3.059e+09,3.06e+09)","[3.06e+09,3.061e+09)","[3.061e+09,3.062e+09)","[3.062e+09,3.063e+09)","[3.063e+09,3.064e+09)","[3.064e+09,3.065e+09)","[3.065e+09,3.066e+09)","[3.095e+09,3.096e+09]"]
    X_cols:list[str] = ["[2.881e+09,2.882e+09)","[2.882e+09,2.883e+09)","[2.883e+09,2.884e+09)","[2.884e+09,2.885e+09)","[2.885e+09,2.886e+09)","[2.886e+09,2.887e+09)","[2.887e+09,2.888e+09)","[2.888e+09,2.889e+09)","[2.889e+09,2.89e+09)","[2.89e+09,2.891e+09)","[2.891e+09,2.892e+09)","[2.892e+09,2.893e+09)","[2.893e+09,2.894e+09)","[2.894e+09,2.895e+09)","[2.895e+09,2.896e+09)","[2.896e+09,2.897e+09)","[2.897e+09,2.898e+09)","[2.898e+09,2.899e+09)","[2.899e+09,2.9e+09)","[2.9e+09,2.901e+09)","[2.901e+09,2.902e+09)","[2.902e+09,2.903e+09)","[2.903e+09,2.904e+09)","[2.904e+09,2.905e+09)","[2.905e+09,2.906e+09)","[2.906e+09,2.907e+09)","[2.907e+09,2.908e+09)","[2.908e+09,2.909e+09)","[2.909e+09,2.91e+09)","[2.91e+09,2.911e+09)","[2.911e+09,2.912e+09)","[2.912e+09,2.913e+09)","[2.913e+09,2.914e+09)","[2.914e+09,2.915e+09)","[2.915e+09,2.916e+09)","[2.916e+09,2.917e+09)","[2.917e+09,2.918e+09)","[2.918e+09,2.919e+09)","[2.919e+09,2.92e+09)","[2.92e+09,2.921e+09)","[2.921e+09,2.922e+09)","[2.922e+09,2.923e+09)","[2.923e+09,2.924e+09)","[2.924e+09,2.925e+09)","[2.925e+09,2.926e+09)","[2.926e+09,2.927e+09)","[2.927e+09,2.928e+09)","[2.928e+09,2.929e+09)","[2.929e+09,2.93e+09)","[2.93e+09,2.931e+09)","[2.931e+09,2.932e+09)","[2.932e+09,2.933e+09)","[2.933e+09,2.934e+09)","[2.934e+09,2.935e+09)","[2.935e+09,2.936e+09)","[2.936e+09,2.937e+09)","[2.937e+09,2.938e+09)","[2.938e+09,2.939e+09)","[2.939e+09,2.94e+09)","[2.942e+09,2.943e+09)","[2.943e+09,2.944e+09)","[2.944e+09,2.945e+09)","[2.945e+09,2.946e+09)","[2.946e+09,2.947e+09)","[2.947e+09,2.948e+09)","[2.948e+09,2.949e+09)","[2.949e+09,2.95e+09)","[2.95e+09,2.951e+09)","[2.951e+09,2.952e+09)","[2.952e+09,2.953e+09)","[2.953e+09,2.954e+09)","[2.954e+09,2.955e+09)","[2.955e+09,2.956e+09)","[2.956e+09,2.957e+09)","[2.957e+09,2.958e+09)","[2.958e+09,2.959e+09)","[2.959e+09,2.96e+09)","[2.96e+09,2.961e+09)","[2.961e+09,2.962e+09)","[2.962e+09,2.963e+09)","[2.963e+09,2.964e+09)","[2.964e+09,2.965e+09)","[2.965e+09,2.966e+09)","[2.966e+09,2.967e+09)","[2.967e+09,2.968e+09)","[2.968e+09,2.969e+09)","[2.969e+09,2.97e+09)","[2.97e+09,2.971e+09)","[2.971e+09,2.972e+09)","[2.972e+09,2.973e+09)","[2.973e+09,2.974e+09)","[2.974e+09,2.975e+09)","[2.975e+09,2.976e+09)","[2.976e+09,2.977e+09)","[2.977e+09,2.978e+09)","[2.978e+09,2.979e+09)","[2.979e+09,2.98e+09)","[2.98e+09,2.981e+09)","[2.981e+09,2.982e+09)","[2.982e+09,2.983e+09)","[2.983e+09,2.984e+09)","[2.984e+09,2.985e+09)","[2.985e+09,2.986e+09)","[2.986e+09,2.987e+09)","[2.987e+09,2.988e+09)","[2.988e+09,2.989e+09)","[2.989e+09,2.99e+09)","[2.99e+09,2.991e+09)","[2.991e+09,2.992e+09)","[2.992e+09,2.993e+09)","[2.993e+09,2.994e+09)","[2.994e+09,2.995e+09)","[2.995e+09,2.996e+09)","[2.996e+09,2.997e+09)","[2.997e+09,2.998e+09)","[2.998e+09,2.999e+09)","[2.999e+09,3e+09)","[3e+09,3.001e+09)","[3.001e+09,3.002e+09)","[3.002e+09,3.003e+09)","[3.003e+09,3.004e+09)","[3.004e+09,3.005e+09)","[3.005e+09,3.006e+09)","[3.006e+09,3.007e+09)","[3.007e+09,3.008e+09)","[3.008e+09,3.009e+09)","[3.009e+09,3.01e+09)","[3.01e+09,3.011e+09)","[3.011e+09,3.012e+09)","[3.012e+09,3.013e+09)","[3.013e+09,3.014e+09)","[3.014e+09,3.015e+09)","[3.015e+09,3.016e+09)","[3.016e+09,3.017e+09)","[3.017e+09,3.018e+09)","[3.018e+09,3.019e+09)","[3.019e+09,3.02e+09)","[3.02e+09,3.021e+09)","[3.021e+09,3.022e+09)","[3.022e+09,3.023e+09)","[3.023e+09,3.024e+09)","[3.024e+09,3.025e+09)","[3.025e+09,3.026e+09)","[3.026e+09,3.027e+09)","[3.027e+09,3.028e+09)","[3.028e+09,3.029e+09)","[3.029e+09,3.03e+09)","[3.03e+09,3.031e+09)","[3.031e+09,3.032e+09)","[3.032e+09,3.033e+09)","[3.033e+09,3.034e+09)","[3.034e+09,3.035e+09)","[3.035e+09,3.036e+09)"]

    def assign_sex_to_profile(genomic_profiles_f:pd.DataFrame, sim_tumors_f:list[str], real_tumors_f:tuple[str], real_sex_f:list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        """
        Function to adapt the genomic profile based on the sex of the donor
        """
        
        # Assign F or M sex based on the simulated genomic profile
        assigned_sex:pd.DataFrame = (genomic_profiles_f[Y_cols]
                                    .assign(id=lambda df: range(0, len(df)))
                                    .melt(id_vars="id", var_name="window", value_name="perc")
                                    .groupby("id", as_index=False)
                                    .agg(perc=("perc", "mean"))
                                    .assign(sex=lambda df: np.where(df["perc"] > 0.000509 + (0.00211 * 3), "M", "F"))[["id", "sex"]]) # 0.000509 mean expression in Females for Y chrom, 0.00211 sd in Females for Y chrom
        assigned_sex['Tumor'] = sim_tumors_f
        expected_sex:pd.DataFrame = pd.DataFrame({'Tumor':real_tumors_f, 'sex':real_sex_f})
        target_sex_counts:pd.DataFrame = (expected_sex.value_counts(["Tumor", "sex"]).reset_index(name="n"))
        sampled_sex_donors:pd.DataFrame = (assigned_sex
                                        .merge(target_sex_counts, on=["Tumor", "sex"], how="inner")
                                        .groupby(["Tumor", "sex"], group_keys=False)
                                        .apply(lambda g: g.sample(n=min(len(g), g["n"].iloc[0]), random_state=42))
                                        .drop(columns="n"))
        sampled_sex_donors["key"] = (sampled_sex_donors["Tumor"].astype(str) + "_" + sampled_sex_donors["sex"].astype(str))
        assigned_ids:list[int] = []
        for _,row in expected_sex.iterrows():
            key:str = f"{row['Tumor']}_{row['sex']}"
            matched_id:pd.Series = sampled_sex_donors.loc[sampled_sex_donors['key'] == key, 'id'].head(n=1)
            assigned_ids.append(matched_id.iloc[0])
            sampled_sex_donors = sampled_sex_donors.drop(matched_id.index)
        
        return (genomic_profiles_f.iloc[assigned_ids].reset_index(drop=True), expected_sex)
    
    def update_sexual_chrom_usage(genomic_profiles_f:pd.DataFrame, exp_genomic_profiles_f:pd.DataFrame, round_genomic_profiles_f:pd.DataFrame, floor_genomic_profiles_f:pd.DataFrame, ceil_genomic_profiles_f:pd.DataFrame, tumor_sex_label_f:pd.DataFrame, sex_ranks_f:pd.DataFrame) -> pd.DataFrame:
    
        """
        Adjust mutation counts per donor so that X/Y usage falls within expected rank intervals.
        """

        def parse_ranks_apply(rank_str:str) -> list[tuple[float, float]]:

            """
            Apply function to parse the rank string
            """
            
            return [tuple(map(float, r.strip("[]").split(","))) for r in rank_str.split(";")]

        def find_or_sample_interval(value:float, intervals:list[tuple[float, float]]) -> tuple[float, float]:
            
            """
            Quick function to find the donor interval regarding sex chromosomes usage
            """

            # Find the interval
            for low, high in intervals:
                if low <= value <= high:
                    return low, high
            
            # If the donor is outside the interval sample one form real stats
            low, high = random.choice(intervals)
            return (low, high)
        
        autosomal_cols:list[str] = [c for c in genomic_profiles_f.columns if c not in X_cols + Y_cols]
        sex_ranks_f["parsed_ranks"] = sex_ranks_f["ranks"].apply(parse_ranks_apply)
        rank_dict:dict = sex_ranks_f.set_index("label")["parsed_ranks"].to_dict()

        profiles:pd.DataFrame = genomic_profiles_f.copy()
        for idx, row in profiles.iterrows():
            ## Set label
            tumor:str = tumor_sex_label_f.loc[idx, "Tumor"]
            if tumor in ['Lymph-MCLL', 'Lymph-UCLL']:
                tumor = 'Lymph-CLL'
            sex:str = tumor_sex_label_f.loc[idx, "sex"]
            label_x:str = f"{tumor}_X{sex}"
            label_y:str = f"{tumor}_Y{sex}"

            ## Compute total counts
            total:int = row.sum()

            ## Calculate autosomal columns only once
            ### Add to autosomal
            add_autosomal_genomic_profiles:pd.Series = ceil_genomic_profiles_f.loc[idx, autosomal_cols] - round_genomic_profiles_f.loc[idx, autosomal_cols]
            add_autosomal_to_half:pd.Series = ceil_genomic_profiles_f.loc[idx, autosomal_cols] - exp_genomic_profiles_f.loc[idx, autosomal_cols]
            add_candidate_autosomal_cols:pd.Index = add_autosomal_genomic_profiles[add_autosomal_genomic_profiles == 1].index
            ### Remove from autosomal
            rm_autosomal_genomic_profiles:pd.Series = round_genomic_profiles_f.loc[idx, autosomal_cols] - floor_genomic_profiles_f.loc[idx, autosomal_cols]
            rm_autosomal_to_half:pd.Series = exp_genomic_profiles_f.loc[idx, autosomal_cols] - floor_genomic_profiles_f.loc[idx, autosomal_cols]
            rm_candidate_autosomal_cols:pd.Index = rm_autosomal_genomic_profiles[rm_autosomal_genomic_profiles == 1].index

            ## Chrom X
            ### Observed frequency and interval
            obs_x_mut:int = row[X_cols].sum()
            obs_x_freq:float = obs_x_mut / total * 100
            low, high = find_or_sample_interval(obs_x_freq, rank_dict[label_x])

            if not (low <= obs_x_freq <= high):
                ### Expected frquency and total mutations
                exp_x_freq:float = np.round(np.random.choice(np.arange(low, high, 0.001), 1)[0], 3)
                exp_x_mut:int = int(np.round(exp_x_freq * total / 100))

                dif_x_mut:int = obs_x_mut - exp_x_mut

                ### More mut than expected -> remove X mutations
                if dif_x_mut > 0:
                    dif_x_genomic_profiles:pd.Series = round_genomic_profiles_f.loc[idx, X_cols] - floor_genomic_profiles_f.loc[idx, X_cols]
                    dif_x_to_half:pd.Series = exp_genomic_profiles_f.loc[idx, X_cols] - floor_genomic_profiles_f.loc[idx, X_cols]
                    selected_autosomal_cols:pd.Index = add_autosomal_to_half.loc[add_candidate_autosomal_cols].sample(frac=1).nsmallest(abs(dif_x_mut)).index
                    add_candidate_autosomal_cols:pd.Index = add_candidate_autosomal_cols.difference(selected_autosomal_cols)
                    x_sign:int = -1
                    autosomal_sign:int = 1

                ### Less mut than expected -> add X mutations
                else:
                    dif_x_genomic_profiles:pd.Series = ceil_genomic_profiles_f.loc[idx, X_cols] - round_genomic_profiles_f.loc[idx, X_cols]
                    dif_x_to_half:pd.Series = ceil_genomic_profiles_f.loc[idx, X_cols] - exp_genomic_profiles_f.loc[idx, X_cols]
                    selected_autosomal_cols:pd.Index = rm_autosomal_to_half.loc[rm_candidate_autosomal_cols].sample(frac=1).nsmallest(abs(dif_x_mut)).index
                    rm_candidate_autosomal_cols:pd.Index = rm_candidate_autosomal_cols.difference(selected_autosomal_cols)
                    x_sign:int = 1
                    autosomal_sign:int = -1

                candidate_x_cols:pd.Index = dif_x_genomic_profiles[dif_x_genomic_profiles == 1].index
                selected_x_cols:pd.Index = dif_x_to_half.loc[candidate_x_cols].sample(frac=1).nsmallest(abs(dif_x_mut)).index
                profiles.loc[idx, selected_x_cols] += x_sign

                n_selected_x_cols:int = len(selected_x_cols)
                while n_selected_x_cols < abs(dif_x_mut):
                    remaining:int = abs(dif_x_mut) - n_selected_x_cols
                    exp_x_row:pd.Series = exp_genomic_profiles_f.loc[idx, X_cols]
                    extra_x_cols:np.ndarray = np.random.choice(exp_x_row[exp_x_row >= np.median(exp_x_row)].index.difference(candidate_x_cols), size=remaining, replace=False)
                    profiles.loc[idx, extra_x_cols] += x_sign
                    n_selected_x_cols += len(extra_x_cols)
                
                profiles.loc[idx, selected_autosomal_cols] += autosomal_sign
            
            ## Chrom Y
            if label_y in rank_dict.keys():
                ### Observed frequency and interval
                obs_y_mut:int = row[Y_cols].sum()
                obs_y_freq:float = obs_y_mut / total * 100
                low, high = find_or_sample_interval(obs_y_freq, rank_dict[label_y])

                if not (low <= obs_y_freq <= high):
                    ### Expected frquency and total mutations
                    exp_y_freq:float = np.round(np.random.choice(np.arange(low, high, 0.001), 1)[0], 3)
                    exp_y_mut:int = int(np.round(exp_y_freq * total / 100))

                    dif_y_mut:int = obs_y_mut - exp_y_mut

                    ### More mut than expected -> remove Y mutations
                    if dif_y_mut > 0:
                        dif_y_genomic_profiles:pd.Series = round_genomic_profiles_f.loc[idx, Y_cols] - floor_genomic_profiles_f.loc[idx, Y_cols]
                        dif_y_to_half:pd.Series = exp_genomic_profiles_f.loc[idx, Y_cols] - floor_genomic_profiles_f.loc[idx, Y_cols]
                        selected_autosomal_cols:pd.Index = add_autosomal_to_half.loc[add_candidate_autosomal_cols].sample(frac=1).nsmallest(abs(dif_y_mut)).index
                        y_sign:int = -1
                        autosomal_sign:int = 1

                    ### Less mut than expected -> add Y mutations
                    else:
                        dif_y_genomic_profiles:pd.Series = ceil_genomic_profiles_f.loc[idx, Y_cols] - round_genomic_profiles_f.loc[idx, Y_cols]
                        dif_y_to_half:pd.Series = ceil_genomic_profiles_f.loc[idx, Y_cols] - exp_genomic_profiles_f.loc[idx, Y_cols]
                        selected_autosomal_cols:pd.Index = rm_autosomal_to_half.loc[rm_candidate_autosomal_cols].sample(frac=1).nsmallest(abs(dif_y_mut)).index
                        y_sign:int = 1
                        autosomal_sign:int = -1

                    candidate_y_cols:pd.Index = dif_y_genomic_profiles[dif_y_genomic_profiles == 1].index
                    selected_y_cols:pd.Index = dif_y_to_half.loc[candidate_y_cols].sample(frac=1).nsmallest(abs(dif_y_mut)).index
                    profiles.loc[idx, selected_y_cols] += y_sign

                    n_selected_y_cols:int = len(selected_y_cols)
                    while n_selected_y_cols < abs(dif_y_mut):
                        remaining:int = abs(dif_y_mut) - n_selected_y_cols
                        exp_y_row:pd.Series = exp_genomic_profiles_f.loc[idx, Y_cols]
                        extra_y_cols:np.ndarray = np.random.choice(exp_y_row[exp_y_row >= np.median(exp_y_row)].index.difference(candidate_y_cols), size=remaining, replace=False)
                        profiles.loc[idx, extra_y_cols] += y_sign
                        n_selected_y_cols += len(extra_y_cols)

                    profiles.loc[idx, selected_autosomal_cols] += autosomal_sign

        profiles = profiles.apply(extra_adjustment_apply, axis=1)
        return profiles

    def extra_adjustment_apply(row:pd.Series) -> pd.Series:

        """
        In very rare cases where only a very small amount of mutations are generated, there might be regions with -1 mutations.
        This function is to fix those regions in a very simple way.
        """

        pos_to_rm:int = abs(row[row < 0].sum())
        if pos_to_rm > 0:
            cols:pd.Index = row.sample(frac=1).nlargest(pos_to_rm).index
            row[cols] -= 1
            row[row < 0] = 0
            return row
        else:
            return row

    # Generate latent profile
    latent_profiles:pd.DataFrame = calo_forest_generation('/oncoGAN/trained_models/positional_pattern', tumor_list_f*5)
    latent_profiles_tumors:list[str] = latent_profiles.pop('Tumor').tolist()

    # Reconstruct the profile
    raw_genomic_profiles:pd.DataFrame = dae_reconstruction(latent_profiles, 'genomic_profile')

    # Assign sex based on the genomic profile
    raw_genomic_profiles, tumor_sex_label = assign_sex_to_profile(raw_genomic_profiles, latent_profiles_tumors, tumor_list_f, sex_f)

    # Clean the output (normalize the row to sum 100%)
    prop_genomic_profiles:pd.DataFrame = raw_genomic_profiles.div(raw_genomic_profiles.sum(axis=1), axis=0)

    # Get numbers instead of percentages
    exp_genomic_profiles:pd.DataFrame = prop_genomic_profiles.multiply(counts_total_f, axis=0)
    round_genomic_profiles:pd.DataFrame = np.round(exp_genomic_profiles).astype(int)
    floor_genomic_profiles:pd.DataFrame = np.floor(exp_genomic_profiles).astype(int)
    ceil_genomic_profiles:pd.DataFrame = np.ceil(exp_genomic_profiles).astype(int)

    # Check that the each profile sums exactly the total number of mutations
    genomic_profiles:pd.DataFrame = round_genomic_profiles.copy()
    for idx, total in enumerate(counts_total_f):
        current:int = int(round_genomic_profiles.iloc[idx].sum())
        missing:int = int(total - current)
        if missing > 0:
            dif_genomic_profiles:pd.Series = ceil_genomic_profiles.iloc[idx] - round_genomic_profiles.iloc[idx]
            dif_to_half :pd.Series= ceil_genomic_profiles.iloc[idx] - exp_genomic_profiles.iloc[idx]
            sign:int = 1
        elif missing < 0:
            dif_genomic_profiles:pd.Series = round_genomic_profiles.iloc[idx] - floor_genomic_profiles.iloc[idx]
            dif_to_half:pd.Series = exp_genomic_profiles.iloc[idx] - floor_genomic_profiles.iloc[idx]
            sign:int = -1
        else:
            continue
        candidate_cols:pd.Index = dif_genomic_profiles[dif_genomic_profiles == 1].index
        selected_cols:pd.Index = dif_to_half.loc[candidate_cols].sample(frac=1).nsmallest(abs(missing)).index
        genomic_profiles.loc[genomic_profiles.index[idx], selected_cols] += sign
    
    # Update sexual chromosomes usage
    tumor_sex_df:pd.DataFrame = pd.read_csv('/oncoGAN/trained_models/xy_usage_ranks.txt', sep='\t')
    genomic_profiles = update_sexual_chrom_usage(genomic_profiles, exp_genomic_profiles, round_genomic_profiles, floor_genomic_profiles, ceil_genomic_profiles,
                                                 tumor_sex_label, tumor_sex_df)
    
    return genomic_profiles

def simulate_driver_profile(tumor_list_f:tuple[str, ...]) -> pd.DataFrame:

    """
    Function to simulate the driver profiles for each donor
    """

    # Simulate the driver profile
    driver_profiles:pd.DataFrame = calo_forest_generation('/oncoGAN/trained_models/driver_profile', tumor_list_f)
    driver_profiles = driver_profiles.drop(columns=['Tumor'])

    # Get numbers instead of percentages
    driver_profiles = driver_profiles.round(0).astype(int)

    return driver_profiles

def simulate_vaf_rank(tumor_list_f:tuple[str, ...]) -> tuple[str, ...]:

    """
    Function to simulate the VAF range for each donor
    """

    rank_file:pd.DataFrame = pd.read_csv("/oncoGAN/trained_models/donor_vaf_rank.tsv", sep='\t') 

    donor_vafs:list[str] = []
    for tumor in tumor_list_f:
        if tumor in ["Lymph-MCLL", "Lymph-UCLL"]:
            tumor = "Lymph-CLL"
        rank_file_i:pd.DataFrame = rank_file.loc[rank_file["tumor"] == tumor]
        vaf:str = random.choices(rank_file_i.columns[1:], weights=rank_file_i.values[0][1:],k=1)[0]
        donor_vafs.append(vaf)

    return tuple(donor_vafs)

def simulate_mut_vafs(tumor_list_f:tuple[str, ...], vaf_ranks_list:tuple[str, ...], counts_total_f:pd.Series, nit_list_f:tuple[float, ...]) -> dict: 

    """
    A function to simulate the VAF of each mutation
    """
    
    def vaf_rank2float(case_mut_vafs_f:list[str]) -> list[float]:

        """
        Convert the VAF rank to a float value
        """

        final_vaf_list:list = []
        for vaf in case_mut_vafs_f:
            vaf:str = re.sub(r'[\[\)\]]', '', vaf)
            start:float = float(vaf.split(',')[0])
            end:float = float(vaf.split(',')[1])
            final_vaf_list.append(round(random.uniform(start, end), 2))

        return(final_vaf_list)

    prop_vaf_file:pd.DataFrame = pd.read_csv(f"/oncoGAN/trained_models/mutation_vaf_rank.tsv", sep='\t')
    mut_vafs:dict = {}
    for idx, (tumor, vaf_rank, n) in enumerate(zip(tumor_list_f, vaf_ranks_list, counts_total_f)):
        if tumor in ["Lymph-MCLL", "Lymph-UCLL"]:
            tumor = "Lymph-CLL"
        case_prop_vaf_file:pd.DataFrame = prop_vaf_file.loc[prop_vaf_file["tumor"]==tumor, ['vaf_range', vaf_rank]]
        case_mut_vafs:list[str] = random.choices(list(case_prop_vaf_file['vaf_range']), weights=list(case_prop_vaf_file[vaf_rank]), k=int(n))
        case_mut_vafs_float:list[float] = vaf_rank2float(case_mut_vafs)
        case_mut_vafs_float_nit:list[float] = [vaf*(1-nit_list_f[idx]) for vaf in case_mut_vafs_float]
        mut_vafs[idx] = case_mut_vafs_float_nit
    
    return(mut_vafs)

########################
# Complete simulations #
########################

def process_chunk(chunk_data:tuple[pd.DataFrame, np.ndarray, np.ndarray], refGenome:str, n_attempt:int) -> pd.DataFrame:
    
    """
    Function to use multi-threading for finding the correct position within simulated positions and their sequences
    """

    def get_sequence(chromosomes_f:np.ndarray, positions_f:np.ndarray, fasta:Fasta, window:int=5000) -> list[str]:
        
        """
        Function to get a DNA sequence around each position
        """

        # Update chrom variable
        chrom_prefix:bool = 'chr1' in fasta.keys()
        if chrom_prefix:
            chromosomes_f = np.char.add("chr", chromosomes_f)
        
        # Define the window
        chrom_sizes:dict = {chrom: len(fasta[chrom]) for chrom in fasta.keys()}
        positions_end:list[int] = []
        for chrom, pos in zip(chromosomes_f, positions_f):
            if pos + window > chrom_sizes[chrom]:
                positions_end.append(chrom_sizes[chrom])
            else:
                positions_end.append(pos + window)

        sequences:list[str] = [fasta[c][s:e].seq for c, s, e in zip(chromosomes_f, positions_f, positions_end)]
        return sequences

    def match_pos2ctx(signatures_f:pd.DataFrame, chromosomes_f:np.ndarray, positions_f:np.ndarray, sequences_f:list) -> pd.DataFrame:

        """
        Function to assign a position to each mutation
        """

        def create_indels_pattern(ref_f:str, alt_f:str, context_f:str) -> tuple[str, int|None]:

            """
            Function to manually create indel patterns. The patterns to be used are located in the ref and alt columns. However, to avoid creating more microhomology than needs to be simulated, we generate all possible contexts that break the extra microhomology
            """

            nucleotides:list[str] = ['A', 'C', 'G', 'T']

            # Exclude the first base from N
            if ':M:' in context_f:
                ref1,ref2 = ref_f.split('|')
                context_length:int = int(context_f.split(':')[-1])
                size:int|None = len(ref1)-int(context_length)
                nt_options1_1:list[str] = [nt for nt in nucleotides if nt != ref1[-(context_length+1)]]
                nt_options1_2:list[str] = [nt for nt in nucleotides if nt != ref1[context_length]]
                nt_options2_1:list[str] = [nt for nt in nucleotides if nt != ref2[-(context_length+1)]]
                nt_options2_2:list[str] = [nt for nt in nucleotides if nt != ref2[context_length]]
                pattern1:list[str] = [f"{n1}{ref1}{n2}" for n1, n2 in itertools.product(nt_options1_1, nt_options1_2)]
                pattern2:list[str] = [f"{n1}{ref2}{n2}" for n1, n2 in itertools.product(nt_options2_1, nt_options2_2)]
                pattern:list[str] = pattern1+pattern2
            else:
                if ref_f != "":
                    nt_options:list[str] = [nt for nt in nucleotides if nt != ref_f[0]]
                else:
                    nt_options:list[str] = [nt for nt in nucleotides if nt != alt_f[0]]
                # Create all combinations: N1 + ref + N2
                pattern:list[str] = [f"{n1}{ref_f}{n2}" for n1, n2 in itertools.product(nt_options, repeat=2)]
                size:int|None = None

            return ("|".join(pattern), size)

        def find_context_in_sequence(signature_i:str, context_i:str, ref_i:str, alt_i:str, sequence_i:str) -> tuple[list[int], list[str]|None, int|None]:

            """
            Function to find the correct context in the DNA sequence
            """

            if signature_i.startswith("SBS"):
                indexes:list[int] = [m.start() for m in re.finditer(context_i, sequence_i)]
                return (indexes, None, None)

            elif signature_i.startswith("ID"):
                refs:list[str] = ref_i.split(",")
                alts:list[str] = alt_i.split(",")

                for r, a in zip(refs, alts):
                    pattern, real_size = create_indels_pattern(r, a, context_i)
                    matches:list[tuple[int, str]] = [(m.start(), m.group()) for m in re.finditer(pattern, sequence_i)]
                    if matches:
                        indexes, patterns = zip(*matches)
                        return (indexes, patterns, real_size)
                return ([], [], None)

            elif signature_i in ("DNP", "TNP", "medium_ins", "big_ins"):
                indexes:list[int] = [m.start() for m in re.finditer(ref_i, sequence_i)]
                return (indexes, None, None)

            elif signature_i in ("medium_del", "big_del"):
                indexes:list[int] = [m.start() for m in re.finditer(alt_i, sequence_i)]
                patterns:list[str] = [sequence_i[i:i+int(ref_i)+1] for i in indexes]
                return (indexes, patterns, None)

            else:
                return ([], None, None)

        def update_mutations(mut_i, position_i:int, ctx_indexes_i:list, indel_patterns_i:list|None, m_size_i:int|None) -> tuple[int|None, str|None, str|None]:

            """
            Function to update the mutation position, ref and alt based on the found context
            """

            class mutation_row_info(NamedTuple):
                signature:str
                contexts:str
                ref:str
                alt:str
                chrom:str|None
                pos:int|None
                updated_ref:str|None
                updated_alt:str|None
            mut_i:mutation_row_info

            if not ctx_indexes_i:
                return (None, None, None)

            index_choice:int = random.randrange(len(ctx_indexes_i))

            # Microhomology INDELs
            if ':M:' in mut_i.contexts and indel_patterns_i is not None and m_size_i is not None:
                m_ref:str = indel_patterns_i[index_choice]

                m_option:int = int(not (mut_i.ref.split('|')[0] in m_ref))

                if m_option == 0:
                    m_context_length:int = int(mut_i.contexts.split(':')[-1])
                    m_pos:int = position_i + ctx_indexes_i[index_choice] + m_context_length + 1
                    ref:str = m_ref[m_context_length:m_context_length + m_size_i + 1]
                    alt:str = ref[0]
                else:
                    m_pos:int = position_i + ctx_indexes_i[index_choice] + len(m_ref) - ((int(m_size_i)+1)) 
                    ref:str = m_ref[-(int(m_size_i)+2):-1]
                    alt:str = ref[0]

            # INDELs
            elif mut_i.signature.startswith('ID') and indel_patterns_i is not None:
                m_pos:int = position_i + ctx_indexes_i[index_choice] + 1
                ref:str = indel_patterns_i[index_choice][:-1]

                if mut_i.alt == "":
                    alt:str = ref[0]
                else:
                    alt_list:list[str] = mut_i.alt.split(',')
                    ref_list:list[str] = mut_i.ref.split(',')
                    alt:str = ref[0] + alt_list[ref_list.index(ref[1:])]

            # SBSs
            elif mut_i.signature.startswith('SBS'):
                m_pos:int = position_i + ctx_indexes_i[index_choice] + 2
                ref:str = mut_i.ref
                alt:str = mut_i.alt

            # DNPs and TNPs
            elif mut_i.signature in ('DNP', 'TNP'):
                m_pos:int = position_i + ctx_indexes_i[index_choice] + 1
                ref:str = mut_i.ref
                alt:str = mut_i.alt
            
            # Medium and big insertions
            elif mut_i.signature in ('medium_ins', 'big_ins'):
                m_pos:int = position_i + ctx_indexes_i[index_choice] + 1
                ref:str = mut_i.ref
                alt:str = ref + mut_i.alt
            
            # Medium and big deletions
            elif mut_i.signature in ('medium_del', 'big_del') and indel_patterns_i is not None:
                m_pos:int = position_i + ctx_indexes_i[index_choice] + 1
                ref:str = indel_patterns_i[index_choice]
                alt:str = mut_i.alt

            else:
                return (None, None, None)

            return (m_pos, ref, alt)
        
        ctx_indexes, indel_patterns, real_m_size = zip(*[find_context_in_sequence(row.signature, row.contexts, row.ref, row.alt, seq) for row, seq in zip(signatures_f.itertuples(index=False), sequences_f)])
        updated_position, updated_ref, updated_alt = zip(*[update_mutations(row, pos, ctx_idxs, id_pat, id_m_size) for row, pos, ctx_idxs, id_pat, id_m_size in zip(signatures_f.itertuples(index=False), positions_f, ctx_indexes, indel_patterns, real_m_size)])
        
        updated_signatures:pd.DataFrame = signatures_f.copy()
        updated_signatures['chrom'] = chromosomes_f
        updated_signatures['pos'] = updated_position
        updated_signatures['updated_ref'] = updated_ref
        updated_signatures['updated_alt'] = updated_alt
        
        return updated_signatures
    
    fasta = Fasta(refGenome)
    
    # Unpack the chunked data
    donor_df_chunk_f, chrom_chunk_f, pos_chunk_f = chunk_data

    # Extract a genomic sequence around each position
    if n_attempt >= 5:
        asgn_sequences_chunk:list[str] = get_sequence(chrom_chunk_f, pos_chunk_f, fasta, window=50000)
    else:    
        asgn_sequences_chunk:list[str] = get_sequence(chrom_chunk_f, pos_chunk_f, fasta)

    # Match the context with the position range
    tmp_donor_df_chunk:pd.DataFrame = match_pos2ctx(donor_df_chunk_f, chrom_chunk_f, pos_chunk_f, asgn_sequences_chunk)
    
    return tmp_donor_df_chunk
    
def select_driver_mutations(tumor_list_f:tuple[str, ...], driver_profile_f:pd.DataFrame) -> dict:
    
    """
    Function to select the driver mutations based on the simulated driver profile"""

    driver_database:pd.DataFrame = pd.read_csv("/oncoGAN/trained_models/driver_profile/driver_mutations_database.csv", delimiter=',')

    # For each donor, get the simulated number of driver mutations from the real dataset
    driver_mutations:dict = {}
    for idx, tumor in enumerate(tumor_list_f):
        case_driver_profile:pd.Series = driver_profile_f.iloc[idx]
        case_driver_mutations:list[pd.DataFrame] = []
        for gene, n in case_driver_profile.items():
            if n <= 0:
                continue
            else:
                driver_muts:pd.DataFrame = driver_database.query("gene_id == @gene and tumor == @tumor")
                if driver_muts.shape[0] == 0:
                    continue
                else:
                    try: # In case we ask to simulate more mutations than the ones available in the database
                        case_driver_mutations.append(driver_muts.sample(n=n, replace=False))
                    except ValueError:
                        case_driver_mutations.append(driver_muts)
        if len(case_driver_mutations) > 0:
            driver_mutations[idx] = pd.concat(case_driver_mutations, ignore_index=True)
        else:
            driver_mutations[idx] = pd.DataFrame(columns=['chrom', 'start', 'mut', 'ref', 'alt', 'gene_id', 'tumor'])
    
    return driver_mutations

def assign_genomic_positions(signatures_f:pd.DataFrame, genomic_pattern_f:pd.Series, refGenome:str, cpus:int) -> pd.DataFrame:
    
    """
    Function to assign genomic positions to each mutation based on the genomic pattern
    """

    def parse_range_map(genomic_interval:str) -> tuple[int, int]:

        """
        Function to parse the range string
        """

        left, right = genomic_interval.strip("[]()").split(",")
        return int(float(left)), int(float(right))
    
    def assign_chromosome(positions_f:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        
        """
        Function to assigned the corresponding chromosome based on a continous position
        """
        
        chromosome_decode:np.array  = np.array(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y'])
        position_decode:np.array  = np.array([0,249250621,492449994,690472424,881626700,1062541960,1233657027,1392795690,1539159712,1680373143,1815907890,1950914406,2084766301,2199936179,2307285719,2409817111,2500171864,2581367074,2659444322,2718573305,2781598825,2829728720,2881033286,3036303846])
        
        position_encode:np.array  = np.digitize(positions_f, position_decode, right=True)
        
        map_positions:np.ndarray = positions_f - np.take(position_decode, position_encode-1)
        map_chromosomes:np.ndarray = np.take(chromosome_decode, position_encode-1)
        
        return (map_chromosomes.astype(str), map_positions.astype(int))
    
    # Randomize the mutational signatures input
    signatures_f = signatures_f.sample(frac=1).reset_index(drop=True)

    # Initialize the output dataframe
    donor_df:pd.DataFrame = signatures_f.copy()
    donor_df["chrom"] = None
    donor_df["pos"] = None
    donor_df["updated_ref"] = None
    donor_df["updated_alt"] = None

    # Define a genomic position for each mutation
    rng = np.random.default_rng()
    position_ranges:np.ndarray = np.array(genomic_pattern_f.index.map(parse_range_map).tolist())
    position_ranges_expanded:np.ndarray = np.repeat(position_ranges, genomic_pattern_f.values, axis=0)

    while_round:int = 0
    n_missing:int = donor_df.shape[0]
    while n_missing > 0 and while_round < 15:
        mask:pd.Series = donor_df["updated_ref"].isna()
        missing_indices:pd.Index = donor_df[mask].index

        # Sample positions only for missing rows
        positions:np.ndarray = np.concatenate([rng.integers(low=lo, high=hi, size=1) for lo, hi in position_ranges_expanded[mask]])
        asgn_chromosomes, asgn_positions = assign_chromosome(positions)
        
        # Prepare multiprocessing
        masked_donor_df:pd.DataFrame = donor_df.loc[mask]
        if masked_donor_df.shape[0] >= cpus and cpus > 1:
            donor_df_chunks:list[pd.DataFrame] = np.array_split(masked_donor_df, cpus)
            chrom_chunks:list[np.ndarray] = np.array_split(asgn_chromosomes, cpus)
            pos_chunks:list[np.ndarray] = np.array_split(asgn_positions, cpus)

            # Create the list of arguments for the worker function
            chunked_args:list[tuple[pd.DataFrame, np.ndarray, np.ndarray]] = list(zip(donor_df_chunks, chrom_chunks, pos_chunks))
            pool_args:list[tuple[tuple[pd.DataFrame, np.ndarray, np.ndarray], str, int]] = [(arg, refGenome, while_round) for arg in chunked_args]

            # Multiprocessing
            with Pool(cpus) as pool:
                results:list[pd.DataFrame] = pool.starmap(process_chunk, pool_args)
            tmp_donor_df:pd.DataFrame = pd.concat(results)
        else:
            tmp_donor_df:pd.DataFrame = process_chunk((masked_donor_df, asgn_chromosomes, asgn_positions), refGenome, while_round)

        tmp_donor_df = tmp_donor_df.reindex(missing_indices)

        # Update only missing rows in the original DataFrame
        donor_df.loc[mask, tmp_donor_df.columns] = tmp_donor_df.values
        
        n_missing = donor_df["updated_ref"].isna().sum()
        while_round += 1

    donor_df = donor_df.dropna()
    donor_df['pos'] = donor_df['pos'].astype(int)
    return donor_df.reset_index(drop=True)

def pd2vcf(muts_f:pd.DataFrame, driver_muts_f:pd.DataFrame, vafs_f:list[float], prefix:str) -> pd.DataFrame:

    """
    Convert the pandas DataFrames into a VCF
    """

    def create_info_field(signatures_list_f:pd.DataFrame, driver_genes_f:pd.Series, vafs_f:list[float]) -> list[str]:
        
        """
        Function to create a VCF info field with more information
        """

        info:list[str] = []
        for row in signatures_list_f.itertuples():
            tmp_info:str = f"AF={vafs_f[row.Index]};MS={row.signature}"

            if 'SBS' in row.signature:
                tmp_info = f"{tmp_info};SBSCTX={row.contexts}"
            elif 'ID' in row.signature:
                tmp_info = f"{tmp_info};IDCTX={row.contexts}"

            if len(row.updated_ref) != 1 and len(row.updated_alt) != 1 and 'Del:M' not in row.contexts:
                tmp_info = f"{tmp_info};HPR={row.updated_ref[1:]}"
            elif 'Del:M' in row.contexts:
                tmp_info = f"{tmp_info};MHR={row.updated_ref[1:]}"

            info.append(tmp_info)
        
        for idx, driver in driver_genes_f.items():
            info.append(f"AF={vafs_f[signatures_list_f.shape[0]+idx]};MS=driver_{driver}")

        return info

    def update_ref_alt_indels_apply(row:pd.Series) -> tuple[str, str]:
        
        """
        Apply function to update the reference and alternative alleles in the correct VCF format
        """

        indel_size:int = len(row['ALT']) - len(row['REF'])
        if (len(row['REF']) != 1 and len(row['ALT']) != 1) and indel_size != 0:
            if indel_size > 0: #INS
                ref:str = row['REF'][0]
                alt:str = row['ALT'][:indel_size+1]
            else: #DEL
                ref:str = row['REF'][:abs(indel_size)+1]
                alt:str = row['ALT'][0]
            return (ref, alt)
        else:
            return (row['REF'], row['ALT'])

    n_muts:int = muts_f.shape[0] + driver_muts_f.shape[0]
    vcf:pd.DataFrame = pd.DataFrame({
        '#CHROM': muts_f['chrom'].tolist() + driver_muts_f['chrom'].tolist(),
        'POS': muts_f['pos'].tolist() + driver_muts_f['start'].tolist(),
        'ID': [prefix] * n_muts,
        'REF': muts_f['updated_ref'].tolist() + driver_muts_f['ref'].tolist(),
        'ALT': muts_f['updated_alt'].tolist() + driver_muts_f['alt'].tolist(),
        'QUAL' : '.',
        'FILTER': '.',
        'INFO': create_info_field(muts_f, driver_muts_f['gene_id'], vafs_f[:n_muts+1])
    })

    # Update REF and ALT fields
    vcf[['REF', 'ALT']] = vcf.apply(update_ref_alt_indels_apply, axis=1, result_type="expand")

    # Sort the VCF
    vcf = vcf.sort_values(by=['#CHROM', 'POS'], key=lambda col: col.map(chrom2int)).reset_index(drop=True)

    # Filter out some random and very infrequent DNP, TNP and repeated SNPs
    vcf['keep'] = abs(vcf['POS'].diff()) > 2
    vcf = vcf[vcf['keep'].shift(-1, fill_value=False)]
    vcf = vcf.drop(columns=['keep']).reset_index(drop=True)

    # Create a unique ID for each mutation
    vcf['INFO'] = vcf.apply(lambda row: f"MUTID={row.name};{row.INFO}", axis=1)

    return(vcf)

######################
# CNA-SV simulations #
######################

def simulate_cna_sv_profile(tumor_list_f:tuple[str, ...], sex_f:list[str], hg19:bool, prefix_list_f:tuple[str, ...]) -> tuple[dict, dict]:
    
    """
    Function to simulate the CNA and SV profiles for each donor based on SimChA tool
    """

    cna:dict = {}
    sv:dict = {}
    for idx, (tumor,sex) in enumerate(zip(tumor_list_f, sex_f)):
        # Create a temp directory to save SimChA results for each donor
        with tempfile.TemporaryDirectory(prefix=f"simcha{idx}_") as tmp_donor_results_dir:
            # Load base CNA model configuration
            cna_model:str = random.choice(default_cna_tumors[tumor])
            cna_model = "main_config" if cna_model == "main" else f"spice_{cna_model}"
            with open(f'/oncoGAN/models/simcha/configs/{cna_model}.json', 'r') as f:
                cna_config:dict = json.load(f)
            
            # Update base CNA configuration
            cna_config['SimParams']['Seed'] = round(time())
            cna_config['SimParams']['Assembly'] = 'hg19' if hg19 else 'hg38'
            cna_config['SimParams']['AutosomesOnly'] = False
            cna_config['SimParams']['Sex'] = 'Male' if sex == 'M' else 'Female'
            
            # Save it in the temp directory
            cna_donor_config_json:str = f'{tmp_donor_results_dir}/simcha{idx}_config.json'
            with open(cna_donor_config_json, 'w') as f:
                json.dump(cna_config, f)

            # Run the CNA simulation in another environment
            command:Sequence[str] = ['dotnet', 'run', '--project', '/oncoGAN/models/simcha', '--',
                                    '--root', '/oncoGAN/models/simcha/',
                                    '--config', f'{cna_donor_config_json}',
                                    '--output', f'{tmp_donor_results_dir}',
                                    '-d', '-s']
            command_logs = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Load results
            donor_cna:pd.DataFrame = pd.read_csv(f'{tmp_donor_results_dir}/copynumbers.tsv', sep='\t').drop(columns=['n_snvs'])
            donor_cna['sample_id'] = prefix_list_f[idx]
            donor_cna['tumor'] = tumor
            donor_sv:pd.DataFrame = pd.read_csv(f'{tmp_donor_results_dir}/events.tsv', sep='\t')
            donor_sv['sample_id'] = prefix_list_f[idx]

            # Return CNA and SV profiles
            cna[idx] = donor_cna
            sv[idx] = donor_sv.rename(columns={"depth":"event_id"})
    
    return (cna, sv)

def update_vafs_cna(refGenome:str, vcf_f:pd.DataFrame, events:pd.DataFrame, nit_f:float) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Function to compute VAF regarding CNA events
    """

    ref_segments = TumorGenome(f"{refGenome}.fai")

    # History definition
    history:list[tuple[str, int]] = [('cna', idx) for idx in events.index]
    for mut in vcf_f.itertuples():
        his_idx:int = random.randint(0, len(history))
        history.insert(his_idx, ('mutation', mut.Index))
    history_df:pd.DataFrame = pd.DataFrame(history, columns=['event', 'event_id']).reset_index(names="event_time")

    # Process history
    for his_type, his_id in history:
        if his_type == 'mutation':
            ref_segments.assign_mutation(his_id, vcf_f.loc[his_id])
        elif his_type == 'cna':
            ref_segments.apply_cna(events.loc[his_id])
    
    # Update VCF with new CNA driven VAFs #TODO - Ignore SVs that do not affect CN status
    new_info_column:list[str] = []
    nit_perc:float = 1.0 - nit_f
    initial_vaf:list[float] = np.random.normal(loc=0.9, scale=0.15, size=vcf_f.shape[0]).tolist()
    for mut in vcf_f.itertuples():
        chrom:str = str(mut._1)
        pos:int = int(mut.POS)
        info_str:str = mut.INFO
        
        cn_mut:int = 0
        cn_total:int = 0
        cna_ids:list[int] = []
        alleles:list[str] = []
        # Calculate cn_total and cn_mut directly from the final graph state
        for seg in ref_segments.segments_dict.values():
            # Count segments that cover the mutation - Total Copy Number
            if seg.contains(chrom, pos):
                cn_total += 1

                # Count segments that contain the mutation - Mutated Copy Number
                if mut.Index in seg.mutations:
                    cn_mut += 1
                    actual_mut:dict = seg.mutations[mut.Index]
                    cna_ids.append(actual_mut['cn'])
                    alleles.append(actual_mut['allele'])

        # Calculate VAF
        m_vaf:float = 0.0
        if cn_total > 0:
            m_vaf = initial_vaf.pop() * (cn_mut / cn_total) * nit_perc

        # VAF boundaries
        m_vaf = m_vaf if m_vaf < 1 else 1 - np.random.normal(loc=0.1, scale=0.03)
        m_vaf = max(0.0, m_vaf)

        # Update INFO field in the VCF
        info_dict:dict = {}
        for item in info_str.split(';'):
            k, v = item.split('=')
            info_dict[k] = v
        info_dict['AF'] = str(round(m_vaf, 2))
        info_dict['TA'] = str(cn_total)
        info_dict['AL'] = ",".join([str(i) for i in cna_ids]) if cna_ids else "."
        
        new_info = []
        for k in ["MUTID", "AF", "TA", "AL", "MS", "SBSCTX", "IDCTX", "HPR", "MHR"]: #Info tags
            if k in info_dict:
                new_info.append(f"{k}={info_dict[k]}")
        new_info_column.append(";".join(new_info))
    
    # Apply the newly built INFO column back to the VCF dataframe
    vcf_f['INFO'] = new_info_column

    # Remove mutations appearing in lost alleles
    af_values:pd.Series = vcf_f["INFO"].str.extract(r'AF=([0-9]*\.?[0-9]+)')[0].astype(float)
    vcf_f = vcf_f[af_values != 0]

    return (vcf_f, history_df)

def plot_cna(cna_f:str, output_f:str) -> None:

    """
    Function to plot the CNA profile of each donor
    """

    command:Sequence[str] = ['Rscript', '/oncoGAN/plot_cna.R',
                             '--cna', cna_f,
                             '--output', output_f]
    plot_result = subprocess.run(command)

@click.group()
def cli():
    pass

@click.command(name="availTumors")
def availTumors(default_tumors_f:list[str]=default_tumors):

    """
    List of available tumors to simulate
    """
    
    available_cna_tumors:list[str] = ["Bladder-TCC", "Breast-AdenoCa", "Cervix-SCC", "CNS-GBM", "CNS-Oligo", "ColoRect-AdenoCA", "Eso-AdenoCa", "Head-SCC", "Kidney-RCC", "Liver-HCC", "Lung-AdenoCA", "Lung-SCC", "Ovary-AdenoCA", "Panc-AdenoCA", "Prost-AdenoCA", "Skin-Melanoma", "Stomach-AdenoCA", "Uterus-AdenoCA"]
    default_tumors_f_cna:list[str] = [f"{tumor}*" if tumor in available_cna_tumors else tumor for tumor in default_tumors_f]
    
    rows:list[str] = []
    for i in range(0, len(default_tumors_f_cna), 6):
        row:list[str] = default_tumors_f_cna[i:i+6]
        formatted_row:str = "".join(f"{tumor:<{18}}" for tumor in row)
        rows.append(formatted_row)
    formatted_tumors:str = "\n".join(rows)

    click.echo(f"\nThis is the list of available tumor types that can be simulated using oncoGAN - ('*' Available CNA model):\n\n{formatted_tumors}\n")

@click.command(name="vcfGANerator")
@click.option("-@", "--cpus",
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of CPUs to use")
@click.option("--tumor",
              type=click.Choice(default_tumors),
              metavar="TEXT",
              show_choices=False,
              default=None,
              show_default=False,
              help="Tumor type to be simulated. Run 'availTumors' subcommand to check the list of available tumors that can be simulated")
@click.option("-n", "--nCases", "nCases",
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of cases to simulate")
@click.option("--NinT", "nit",
              type=click.FLOAT,
              default=0.0,
              show_default=True,
              help="Normal in Tumor contamination to be taken into account when adjusting VAF for CNA-SV events (e.g. 0.20 = 20%)")
@click.option("--template", "template",
              type=click.Path(exists=True, file_okay=True),
              default=None,
              show_default=False,
              help="File in CSV format with the number of each type of mutation to simulate for each donor (template available on GitHub)")
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
@click.option("--hg19", "hg19",
              is_flag=True,
              required=False,
              help="Transform the mutations to hg19. Default hg38")
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
def oncoGAN(cpus, tumor, nCases, nit, template, refGenome, prefix, outDir, hg19, simulateMuts, simulateCNA_SV, savePlots):

    """
    Command to simulate mutations (VCF), CNAs and SVs for different tumor types using a Flow-Matching Diffusion model
    """
    
    # set_start_method("spawn", force=True) #For debugging in VSCode
    
    # Check that CLI parameters are correct
    if tumor is None and template is None:
        raise click.UsageError("You must provide either a tumor type using the '--tumor' option or a template using the '--template' option. Run 'availTumors' subcommand to check the list of available tumors that can be simulated.")
    if tumor is not None and template is not None:
        raise click.UsageError("You cannot provide both a tumor type using the '--tumor' option and a template using the '--template' option at the same time. Please choose one of the two options. Run 'availTumors' subcommand to check the list of available tumors that can be simulated.")

    # Create the output directory if it doesn't exist
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    
    # Simulate counts for each type of mutation
    if template is None:
        counts:pd.DataFrame = simulate_counts(tumor, nCases)
        prefix_list:tuple[str, ...] = tuple(f"{(prefix or 'sim')}{idx+1}" for idx in range(nCases))
        nit_list:tuple[float, ...] = tuple(nit for _ in range(nCases))
        counts_tumor_tag:tuple[str, ...] = tuple(counts.pop('Tumor').to_list())
        counts_total:pd.Series = counts.sum(axis=1).astype(int)
    else:
        counts:pd.DataFrame = validate_template(template, default_tumors)
        prefix_list:tuple[str, ...] = tuple(counts.pop('ID').to_list())
        nit_list:tuple[float, ...] = tuple(counts.pop('NinT').to_list())
        counts_tumor_tag:tuple[str, ...] = tuple(counts.pop('Tumor').to_list())
        counts_total:pd.Series = counts.sum(axis=1).astype(int)

    # Simulate sex
    sex:list[str] = simulate_sex(counts_tumor_tag)
    
    if simulateMuts:
        # Simulate mutational signatures (SBS and ID)
        signatures:dict = simulate_signatures(counts)
        
        # Simulate genomic pattern profiles
        genomic_patterns:pd.DataFrame = simulate_genomic_profile(counts_tumor_tag, counts_total, sex)

        # Simulate driver profiles
        driver_profiles:pd.DataFrame = simulate_driver_profile(counts_tumor_tag)
        driver_mutations:dict = select_driver_mutations(counts_tumor_tag, driver_profiles)
        counts_drivers_total:pd.Series = counts_total + driver_profiles.sum(axis=1)
        
        # Simulate VAFs or leave them empty to be simulated depending on CNA status
        mut_vafs:dict = {}
        if not simulateCNA_SV:
            # Simulate donor and mutations VAFs
            donor_vaf_ranks:tuple[str, ...] = simulate_vaf_rank(counts_tumor_tag)
            mut_vafs = simulate_mut_vafs(counts_tumor_tag, donor_vaf_ranks, counts_drivers_total, nit_list)
        else:
            mut_vafs = {idx: [0.0] * counts_drivers_total[idx] for idx in range(len(counts_tumor_tag))}
            
    if simulateCNA_SV:
        # Simulate CNA and SV profiles
        cna_profile, sv_profile = simulate_cna_sv_profile(counts_tumor_tag, sex, hg19, prefix_list)

    # Assemble one donor at a time
    for idx, case_tumor in tqdm(enumerate(counts_tumor_tag), desc = "Donors"):
        output:str = f"{outDir}/{tumor}_{prefix_list[idx]}"
        
        if simulateMuts:
            case_signatures:pd.DataFrame = signatures[idx].reset_index(drop=True)
            case_genomic_pattern:pd.Series = genomic_patterns.iloc[idx]
            case_driver_mutations:pd.Series = driver_mutations[idx].reset_index(drop=True)
            case_mut_vafs:list[float] = mut_vafs[idx]

            # Generate the chromosome and position of the mutations
            case_genomic_positions:pd.DataFrame = assign_genomic_positions(case_signatures, case_genomic_pattern, refGenome, cpus)
        
            # Create the VCF output 
            case_vcf:pd.DataFrame = pd2vcf(case_genomic_positions, case_driver_mutations, case_mut_vafs, prefix=prefix_list[idx])
            
            # Convert from hg19 to hg38
            if not hg19:
                case_vcf = hg19tohg38(case_vcf)

        if simulateCNA_SV:
            case_cna:pd.DataFrame = cna_profile[idx]
            case_sv:pd.DataFrame = sv_profile[idx]

            if simulateMuts:
                case_nit:float = nit_list[idx]
                case_vcf, case_event_history = update_vafs_cna(refGenome, case_vcf, case_sv, case_nit)
        
        # Write the outputs
        if simulateMuts and simulateCNA_SV:
            with open(f"{output}.vcf", "w+") as out:
                out.write("##fileformat=VCFv4.2\n")
                out.write(f"##fileDate={date.today().strftime('%Y%m%d')}\n")
                out.write(f"##source=OncoGAN-v{VERSION}\n")
                out.write(f"##reference={'hg19' if hg19 else 'hg38'}\n")
                out.write('##INFO=<ID=MUTID,Number=A,Type=Integer,Description="Mutation ID">\n')
                out.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">\n')
                out.write('##INFO=<ID=TA,Number=A,Type=Integer,Description="Total number of alleles in which the mutation can appear">\n')
                out.write('##INFO=<ID=AL,Number=A,Type=String,Description="Copy Number IDs in which the mutation is located">\n')
                out.write('##INFO=<ID=MS,Number=A,Type=String,Description="Mutation type or mutational signature assigned to each mutation. Available options are: SBS (single base substitution signature), DNP (dinucleotide polymorphism), TNP (trinucleotide polymorphism), ID (indel signature), driver_* (driver mutation sampled from real donors), medium_ins/del (>5 indel size <=10), big_ins/del (>10 indel size <=25)">\n')
                out.write('##INFO=<ID=SBSCTX,Number=A,Type=String,Description="SBS96 context">\n')
                out.write('##INFO=<ID=IDCTX,Number=A,Type=String,Description="Indel context">\n')
                out.write('##INFO=<ID=HPR,Number=A,Type=String,Description="Homopolymer reference">\n')
                out.write('##INFO=<ID=MHR,Number=A,Type=String,Description="Microhomology reference">\n')
            case_vcf.to_csv(f"{output}.vcf", sep="\t", index=False, mode = "a")
            case_cna.to_csv(f"{output}_cna.tsv", sep="\t", index=False)
            case_sv.to_csv(f"{output}_sv.tsv", sep="\t", index=False)
            case_event_history.to_csv(f"{output}_events_order.tsv", sep="\t", index=False)
        elif simulateMuts and not simulateCNA_SV:
            with open(f"{output}.vcf", "w+") as out:
                out.write("##fileformat=VCFv4.2\n")
                out.write(f"##fileDate={date.today().strftime('%Y%m%d')}\n")
                out.write(f"##source=OncoGAN-v{VERSION}\n")
                out.write(f"##reference={'hg19' if hg19 else 'hg38'}\n")
                out.write('##INFO=<ID=MUTID,Number=A,Type=Integer,Description="Mutation ID">\n')
                out.write('##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">\n')
                out.write('##INFO=<ID=MS,Number=A,Type=String,Description="Mutation type or mutational signature assigned to each mutation. Available options are: SBS (single base substitution signature), DNP (dinucleotide polymorphism), TNP (trinucleotide polymorphism), ID (indel signature), driver_* (driver mutation sampled from real donors), medium_ins/del (>5 indel size <=10), big_ins/del (>10 indel size <=25)">\n')
                out.write('##INFO=<ID=SBSCTX,Number=A,Type=String,Description="SBS96 context">\n')
                out.write('##INFO=<ID=IDCTX,Number=A,Type=String,Description="Indel context">\n')
                out.write('##INFO=<ID=HPR,Number=A,Type=String,Description="Homopolymer reference">\n')
                out.write('##INFO=<ID=MHR,Number=A,Type=String,Description="Microhomology reference">\n')
            case_vcf.to_csv(f"{output}.vcf", sep="\t", index=False)
        elif not simulateMuts and simulateCNA_SV:
            case_cna.to_csv(f"{output}_cna.tsv", sep="\t", index=False)
            case_sv.to_csv(f"{output}_sv.tsv", sep="\t", index=False)
        
        # Plot CNA profile
        if simulateCNA_SV and savePlots:
            plot_cna(f"{output}_cna.tsv", f"{output}_cna_profile.png")
