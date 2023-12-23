import os
import re
import pandas as pd
import allel
from Bio.Seq import reverse_complement
from liftover import get_lifter
from pyfaidx import Fasta

def hg38tohg19(vcf:pd.DataFrame) -> pd.DataFrame:

    """
    Convert hg38 coordinates to hg19
    """

    converter = get_lifter('hg38', 'hg19')
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

def vcf2df(vcf:os.path, prefix:bool, liftOver:bool) -> pd.DataFrame:
    
    """
    Filter SNVs in chr1-chr22 from VCF file and return a dataframe
    """

    # Open VCF
    vcf:pd.DataFrame = allel.vcf_to_dataframe(vcf, fields='*', alt_number=2)

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

    # Filter SNVs in chr1-chr22
    vcf_filter:pd.DataFrame = vcf[(vcf['is_snp'] == True) & (vcf['CHROM'].isin(chr_list))]
    print(f"\n--- {len(vcf_filter)} SNVs from a total of {len(vcf)} variants in the VCF file ---\n")

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
    df_bins:pd.Series = df.CHROM + '.' + df.POS.apply(lambda x: int(round(float(x) / 1000000))).astype(str)
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

    # Extract the mutation types
    changes:list = []
    for _,row in df.iterrows():
        chrom:str = str(row['CHROM'])
        pos:int = int(row['POS'])
        ref:str = row['REF']
        alt:str = row['ALT_1']
        ref_ctx:str = fasta[chrom][pos-2:pos+1].seq.upper()

        # Check that we have the same reference bases
        if (ref != ref_ctx[1]):
            print('-----------------------------------')
            print("WARNING: Reference base from VCF file doesn't match with records on the provided reference genome")
            print(f'{chrom}:{pos} -- VCF: {ref} vs Reference genome: {ref_ctx[1]} -- Reference context: {ref_ctx}')
            print('-----------------------------------')
            continue #TODO - Ask Wei why he keeps these mutations

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