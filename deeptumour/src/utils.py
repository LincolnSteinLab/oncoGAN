import os
import re
import pickle
import pandas as pd
import allel
import math
from Bio.Seq import reverse_complement
from liftover import ChainFile
from pyfaidx import Fasta

def hg38tohg19(vcf:pd.DataFrame) -> pd.DataFrame:

    """
    Convert hg38 coordinates to hg19
    """

    converter = ChainFile('/.liftover/hg38ToHg19.over.chain.gz')
    for i,row in vcf.iterrows():
        chrom:str = str(row['CHROM'])
        pos:int = int(row['POS'])
        try:
            liftOver_result:tuple = converter[chrom][pos][0]
            vcf.loc[i, 'CHROM'] = liftOver_result[0]
            vcf.loc[i, 'POS'] = liftOver_result[1]
        except IndexError:
            vcf.loc[i, 'CHROM'] = 'Remove'

    return(vcf)

def process_dnp(vcf:pd.DataFrame) -> pd.DataFrame:

    """
    Split DNPs into two different SNPs
    """
    
    # Select DNPs positions
    dnps = vcf[(vcf['REF'].str.len() == 2) & (vcf['ALT'].str.len() == 2)]

    # Duplicate the DNPs and create the second SNP
    dnps_dup = dnps.copy()
    dnps_dup['REF'] = dnps_dup['REF'].str[1]
    dnps_dup['ALT'] = dnps_dup['ALT'].str[1]
    dnps_dup['POS'] = dnps_dup['POS'] + 1
    dnps_dup['is_snp'] = True

    # Modify the original DNPs to create the first SNP
    vcf.loc[dnps.index, 'REF'] = dnps['REF'].str[0]
    vcf.loc[dnps.index, 'ALT'] = dnps['ALT'].str[0]
    vcf.loc[dnps.index, 'is_snp'] = True

    # Concatenate the original df with the duplicated rows
    new_vcf = pd.concat([vcf, dnps_dup], ignore_index=True).sort_values(by=['CHROM', 'POS'])

    return(new_vcf.reset_index(drop=True))

def process_tnp(vcf:pd.DataFrame) -> pd.DataFrame:

    """
    Split TNPs into three different SNPs
    """
    
    # Select TNPs positions
    tnps = vcf[(vcf['REF'].str.len() == 3) & (vcf['ALT'].str.len() == 3)]

    # Duplicate the TNPs and create the second SNP
    tnps_dup_second = tnps.copy()
    tnps_dup_second['REF'] = tnps_dup_second['REF'].str[1]
    tnps_dup_second['ALT'] = tnps_dup_second['ALT'].str[1]
    tnps_dup_second['POS'] = tnps_dup_second['POS'] + 1
    tnps_dup_second['is_snp'] = True

    # Duplicate the TNPs and create the third SNP
    tnps_dup_third = tnps.copy()
    tnps_dup_third['REF'] = tnps_dup_third['REF'].str[2]
    tnps_dup_third['ALT'] = tnps_dup_third['ALT'].str[2]
    tnps_dup_third['POS'] = tnps_dup_third['POS'] + 2 
    tnps_dup_third['is_snp'] = True

    # Modify the original TNPs to create the first SNP
    vcf.loc[tnps.index, 'REF'] = tnps['REF'].str[0]
    vcf.loc[tnps.index, 'ALT'] = tnps['ALT'].str[0]
    vcf.loc[tnps.index, 'is_snp'] = True

    # Concatenate the original df with the duplicated rows
    new_vcf = pd.concat([vcf, tnps_dup_second, tnps_dup_third], ignore_index=True).sort_values(by=['CHROM', 'POS'])

    return(new_vcf.reset_index(drop=True))

def vcf2df(vcf:os.path, prefix:bool, liftOver:bool) -> pd.DataFrame:
    
    """
    Filter SNVs in chr1-chr22 from VCF file and return a dataframe
    """

    # Open VCF
    vcf:pd.DataFrame = allel.vcf_to_dataframe(vcf, fields='*', alt_number=1)
    vcf.drop_duplicates(inplace=True)
    vcf.reset_index(drop=True, inplace=True)

    # LiftOver coordinates if the original VCF is in hg38
    if liftOver:
        vcf = hg38tohg19(vcf)
    
    # Select chromosomes
    if prefix:
        chr_list:list = [f"chr{str(chrom)}" for chrom in range(1, 23)]
    else:
        chr_list:list = [str(chrom) for chrom in range(1, 23)]

    # Update chromosome names
    if prefix and not (vcf['CHROM'][0].startswith('chr')):
        vcf['CHROM'] = [f"chr{str(chrom)}" for chrom in vcf['CHROM']]
    elif not prefix and (vcf['CHROM'][0].startswith('chr')):
        vcf['CHROM'] = [str(chrom).replace('chr', '') for chrom in vcf['CHROM']]
    else:
        pass

    # Split DNPs and TNPs
    vcf = process_dnp(vcf)
    vcf = process_tnp(vcf)

    # Filter SNVs in chr1-chr22
    vcf_filter:pd.DataFrame = vcf[(vcf['is_snp'] == True) & (vcf['CHROM'].isin(chr_list)) & (vcf['REF'] != '-') & (vcf['ALT'] != '-')]

    return(vcf_filter.reset_index(drop=True))

def df2bins(df:pd.DataFrame, sample_name:str, prefix:bool) -> pd.DataFrame:

    """
    Convert the dataframe to bin counts
    """
    
    # Load the header of the bins
    header_bins:pd.DataFrame = pd.read_csv('/DeepTumour/trained_models/hg19.1Mb.header.gz', compression='gzip', header=None)

    # Update chromosome names
    if not prefix:
        header_bins.iloc[:, 0] = header_bins.iloc[:, 0].apply(lambda x: str(x).replace('chr', ''))

    # Get bins from the df
    df_bins:pd.Series = df.CHROM + '.' + df.POS.apply(lambda x: int(math.floor(float(x) / 1000000))).astype(str)
    bins:pd.DataFrame = pd.DataFrame({'bins': pd.Series(pd.Categorical(df_bins, categories=header_bins.iloc[:, 0]))})
    
    # Group bins and count
    bins = bins.groupby('bins').size().reset_index(name=sample_name)

    return(bins)

def df2mut(df:pd.DataFrame, sample_name:str, fasta:Fasta) -> pd.DataFrame:

    """
    Convert the dataframe to mutation types
    """

    # Load the header of the mutation types
    header_muts:pd.DataFrame = pd.read_csv('/DeepTumour/trained_models/Mut-Type-Header.csv')

    # Load Z-Norm parameters
    with open("/DeepTumour/trained_models/z-norm.pkl", "rb") as f:
        z_norm:dict = pickle.load(f)

    # Extract the mutation types
    changes:list = []
    for _,row in df.iterrows():
        chrom:str = str(row['CHROM'])
        pos:int = int(row['POS'])
        ref:str = row['REF']
        alt:str = row['ALT']
        ref_ctx:str = fasta[chrom][pos-2:pos+1].seq.upper()

        # Check that we have the same reference bases
        if (ref != ref_ctx[1]):
            print('-----------------------------------')
            print("WARNING: Reference base from VCF file doesn't match with records on the provided reference genome")
            print(f'{chrom}:{pos} -- VCF: {ref} vs Reference genome: {ref_ctx[1]} -- Reference context: {ref_ctx}')
            print('-----------------------------------')
            continue

        # Get the reverse complement if necessary
        if (re.search('[GT]', ref)):
            ref = reverse_complement(ref)
            alt = reverse_complement(alt)
            ref_ctx = reverse_complement(ref_ctx)

        # Calculate the mutation types
        ## Single context
        changes.append(f'{ref}..{alt}')
        ## Binucleotide context
        changes.append(f'{ref_ctx[:-1]}..{ref_ctx[0]}{alt}')
        changes.append(f'{ref_ctx[1:]}..{alt}{ref_ctx[-1]}')
        ## Trinucleotide context
        changes.append(f'{ref_ctx}..{ref_ctx[0]}{alt}{ref_ctx[-1]}')

    # Group mutation types and count
    mutations:pd.DataFrame = pd.DataFrame({"bins": pd.Series(pd.Categorical(changes, categories=header_muts.iloc[:, 0]))})
    mutations = mutations.groupby('bins').size().reset_index(name=sample_name)
    # Calculate proportions for each range
    if sum(mutations[sample_name]) > 0:
        sgl_prop = mutations[sample_name].iloc[0:6] / mutations[sample_name].iloc[0:6].sum()
        di_prop = mutations[sample_name].iloc[6:54] / mutations[sample_name].iloc[6:54].sum()
        tri_prop = mutations[sample_name].iloc[54:150] / mutations[sample_name].iloc[54:150].sum()
        mutations.loc[0:5, sample_name] = sgl_prop
        mutations.loc[6:53, sample_name] = di_prop
        mutations.loc[54:149, sample_name] = tri_prop

        ## z-norm
        mutations[mutations.columns[1:]] = mutations.apply(lambda x: (x[1:] - z_norm[x['bins']]['mean']) / z_norm[x['bins']]['std'], axis=1)

    return(mutations)

def vcf2input(vcf:os.path, refGenome:os.path, liftOver:bool) -> pd.DataFrame:

    """
    Process the VCF to get the input necessary for DeepTumour
    """

    # Create output name
    sample_name:str = os.path.basename(vcf).replace('.vcf', '')

    # Load the reference genome
    fasta:Fasta = Fasta(refGenome)
    prefix:bool = list(fasta.keys())[0].startswith('chr')

    # Load the VCF
    df:pd.DataFrame = vcf2df(vcf, prefix, liftOver)

    # Convert the dataframe to bin counts
    bins:pd.DataFrame = df2bins(df, sample_name, prefix)

    # Convert the dataframe to mutation types
    mutations:pd.DataFrame = df2mut(df, sample_name, fasta)

    # Merge the dataframes
    input:pd.DataFrame = pd.concat([bins, mutations]).set_index('bins')
    input = input.transpose()
    input.reset_index(drop=False, inplace=True)
    
    return(input)