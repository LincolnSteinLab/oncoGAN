import os
import glob
import click
import subprocess
import random
import pandas as pd
from pyfaidx import Fasta

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

def readVCF(input:click.Path, dbSNP:bool = False) -> pd.DataFrame:

    """
    A function to open the VCF as a pd DataFrame
    """
    
    vcf:pd.DataFrame = pd.read_table(input, sep = "\t", comment = '#',
                        names = ['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'])
    vcf = vcf.sort_values(by=['chrom', 'pos'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)
    
    if dbSNP:
        vcf = vcf.drop(columns=['qual', 'filter', 'info'])
    else:
        vcf['snv_id'] = vcf['id'].apply(lambda x: x.split('_')[0])
        vcf['id'] = vcf['id'].apply(lambda x: x.split('_')[1])
        vcf[['af', 'ms', 'ta', 'al', 'cn']] = vcf['info'].str.extract(r'AF=([\d.]+);MS=([^;]+);TA=([^;]+);AL=([^;]+);CN=([^;]+)')
        vcf['af'] = vcf['af'].astype(float)
        vcf['ta'] = vcf['ta'].astype(int)
        vcf = vcf.drop(columns=['qual', 'filter', 'info', 'ms'])
    
    return vcf

def introduce_polymorphisms(genome:Fasta, dbsnp:pd.DataFrame) -> tuple:

    """
    Add polymorphisms to the reference genome
    """

    # Randomly assign haplotypes for each SNP
    dbsnp['allele'] = random.choices(['allele_1_minor', 'allele_2_major', 'homozygous'], k = dbsnp.shape[0])

    updated_genome:dict = {}
    updated_positions:pd.DataFrame = pd.DataFrame()
    for chrom in genome.keys():
        for allele in ['allele_1_minor', 'allele_2_major']:
            mov:int = 0
            error_muts:pd.DataFrame = pd.DataFrame() 
            updated_chrom:str = str(genome[chrom])

            chrom_dbsnp:pd.DataFrame = dbsnp[(dbsnp['chrom'].astype('str') == str(chrom)) & (dbsnp['allele'].isin([allele, 'homozygous']))]
            for _,mut in chrom_dbsnp.iterrows():
                position = mut['pos']+mov-1
                if ((len(mut['ref']) == 1) and (len(mut['alt']) == 1)): #SNP
                    if mut['ref'][0] != updated_chrom[position]:
                        error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                        continue
                    else:
                        updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+1:]
                        continue
                elif ((len(mut['ref']) == 2) and (len(mut['alt']) == 2)): #DNP
                    if mut['ref'] != updated_chrom[position:position+2]:
                        error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                        continue
                    else:
                        updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+2:]
                        continue
                elif ((len(mut['ref']) == 3) and (len(mut['alt']) == 3)): #TNP
                    if mut['ref'][0] != updated_chrom[position:position+3]:
                        error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                        continue
                    else:
                        updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+3:]
                        continue
                elif (len(mut['ref']) > 1): #DEL
                    if mut['ref'][0] != updated_chrom[position]:
                        error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                        continue
                    else:
                        updated_chrom = updated_chrom[:position+1] + updated_chrom[position+len(mut['ref']):]
                        mov -= len(mut['ref'])-1
                        updated_positions = pd.concat([updated_positions, pd.DataFrame(data={'chrom': [chrom], 'pos': [mut['pos']], 'allele': [allele], 'mov': [mov]})])
                        continue
                elif (len(mut['alt']) > 1): #INS
                    if mut['ref'][0] != updated_chrom[position]:
                        error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                        continue
                    else:
                        updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+1:]
                        mov += len(mut['alt'])-1
                        updated_positions = pd.concat([updated_positions, pd.DataFrame(data={'chrom': [chrom], 'pos': [mut['pos']], 'allele': [allele], 'mov': [mov]})])
                        continue

            updated_genome[f'{chrom}_{allele}'] = updated_chrom
            updated_positions = updated_positions.reset_index(drop=True)

    return(updated_genome, dbsnp, updated_positions)

def initialize_genome(genome:Fasta) -> tuple:

    """
    Initialize the genome with the two alleles
    """

    updated_genome:dict = {}
    updated_positions:pd.DataFrame = pd.DataFrame()
    for chrom in genome.keys():
        updated_genome[f'{chrom}_allele_2_major'] = str(genome[chrom])
        updated_genome[f'{chrom}_allele_1_minor'] = str(genome[chrom])
        updated_positions = pd.concat([updated_positions, pd.DataFrame(data={'chrom': [chrom], 'pos': [0], 'allele': ['allele_2_major'], 'mov': [0]})])
        updated_positions = pd.concat([updated_positions, pd.DataFrame(data={'chrom': [chrom], 'pos': [0], 'allele': ['allele_1_minor'], 'mov': [0]})])

    return(updated_genome, updated_positions)

def get_mov(info_df:pd.DataFrame, mut:pd.Series, allele) -> int:

    """
    Keep the trace of the length difference of the custom genome with respect to the reference genome
    """

    info_df_filtered:pd.DataFrame = info_df[(info_df['chrom'].astype(str) == str(mut['chrom'])) & (info_df['pos'] < mut['pos']) & (info_df['allele'] == allele)]
    if info_df_filtered.empty:
        return 0
    else:
        last_row:pd.Series = info_df_filtered.tail(1).iloc[0]
        return last_row['mov']

def update_next_movs(info_df:pd.DataFrame, chrom:str, allele:str, pos:int, mov2up:int) -> pd.DataFrame:
    
    """
    Update changes in the reference genome length
    """

    muts2up:pd.Index = info_df[(info_df['chrom'].astype(str) == str(chrom)) & (info_df['allele'] == allele) & (info_df['pos'].astype(int) > pos)].index
    info_df.loc[muts2up, 'mov'] += mov2up

    return(info_df)

def get_sv_mov(germ_info:pd.DataFrame, somatic_info:pd.DataFrame, sv_info:pd.DataFrame, chrom:str, start:int, end:int, allele:str) -> tuple:

    """
    Helper to extract the updated coordinates for SVs
    """

    germ_mov:int = get_mov(germ_info, pd.Series(data=[chrom, start], index=['chrom', 'pos']), allele)
    somatic_mov:int = get_mov(somatic_info, pd.Series(data=[chrom, start], index=['chrom', 'pos']), allele)
    sv_mov:int = get_mov(sv_info, pd.Series(data=[chrom, start], index=['chrom', 'pos']), allele)
    start:int = start+sv_mov+somatic_mov+germ_mov-1

    germ_mov:int = get_mov(germ_info, pd.Series(data=[chrom, end], index=['chrom', 'pos']), allele)
    somatic_mov:int = get_mov(somatic_info, pd.Series(data=[chrom, end], index=['chrom', 'pos']), allele)
    sv_mov:int = get_mov(sv_info, pd.Series(data=[chrom, end], index=['chrom', 'pos']), allele)
    end:int = end+sv_mov+somatic_mov+germ_mov-1

    return(start, end, sv_mov)

def introduce_mutations(genome:dict, mutations:pd.DataFrame, germ_info:pd.DataFrame, events:click.Path, svs:click.Path, outDir:click.Path, donor_id:str) -> pd.DataFrame:

    """
    Add mutations to the custom reference genome
    """

    # Open event and CNA files
    events_df:pd.DataFrame = pd.read_csv(events, delimiter='\t')
    sv_df:pd.DataFrame = pd.read_csv(svs, delimiter='\t')

    # Follow the order of events to introduce the mutations and CNAs
    somatic_info:pd.DataFrame = pd.DataFrame(columns=['chrom', 'pos', 'allele', 'mov'])
    for _,event in events_df.iterrows():
        if event['class'] == "MUT":
            try:
                mut:pd.Series = mutations[mutations['snv_id'] == event['event_id']].iloc[0]
            except IndexError:
                ## In case the mutation is located in a deleted allele
                continue

            ## Extract position modifiers
            if germ_info is None:
                germ_mov:int = 0
            else:
                germ_mov:int = get_mov(germ_info, mut, event['allele'])
            somatic_mov:int = get_mov(somatic_info, mut, event['allele'])
            position:int = mut['pos']+somatic_mov+germ_mov-1

            ## Add the mutations
            chrom_key:str = f"{mut['chrom']}_{event['allele']}"        
            updated_chrom:str = genome[chrom_key]
            if ((len(mut['ref']) == 1) and (len(mut['alt']) == 1)): #SNP
                if mut['ref'][0] != updated_chrom[position]:
                    continue
                else:
                    updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+1:]
                    genome[chrom_key] = updated_chrom
                    continue
            elif ((len(mut['ref']) == 2) and (len(mut['alt']) == 2)): #DNP
                if mut['ref'] != updated_chrom[position:position+2]:
                    continue
                else:
                    updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+2:]
                    genome[chrom_key] = updated_chrom
                    continue
            elif ((len(mut['ref']) == 3) and (len(mut['alt']) == 3)): #TNP
                if mut['ref'][0] != updated_chrom[position:position+3]:
                    continue
                else:
                    updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+3:]
                    genome[chrom_key] = updated_chrom
                    continue
            elif (len(mut['ref']) > 1): #DEL
                if mut['ref'][0] != updated_chrom[position]:
                    continue
                else:
                    updated_chrom = updated_chrom[:position+1] + updated_chrom[position+len(mut['ref']):]
                    genome[chrom_key] = updated_chrom
                    somatic_mov -= len(mut['ref'])-1
                    somatic_info = pd.concat([somatic_info, pd.DataFrame(data={'chrom': [mut['chrom']], 'pos': [mut['pos']], 'allele': [event['allele']], 'mov': [somatic_mov]})])
                    somatic_info = somatic_info.sort_values(by=['chrom', 'pos'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)
                    somatic_info = update_next_movs(somatic_info, mut['chrom'], event['allele'], mut['pos'], -(len(mut['ref'])-1))
                    continue
            elif (len(mut['alt']) > 1): #INS
                if mut['ref'][0] != updated_chrom[position]:
                    continue
                else:
                    updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+1:]
                    genome[chrom_key] = updated_chrom
                    somatic_mov += len(mut['alt'])-1
                    somatic_info = pd.concat([somatic_info, pd.DataFrame(data={'chrom': [mut['chrom']], 'pos': [mut['pos']], 'allele': [event['allele']], 'mov': [somatic_mov]})])
                    somatic_info = somatic_info.sort_values(by=['chrom', 'pos'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)
                    somatic_info = update_next_movs(somatic_info, mut['chrom'], event['allele'], mut['pos'], len(mut['alt'])-1)
                    continue
        elif event['class'] == "DEL":
            ## Deletions are processed later
            continue
        elif event['class'] == "DUP":
            ## Create a new allele for the duplication
            chrom:str = str(event['cna_id'].split('_')[0].replace('x', ''))
            chrom_key:str = f"{chrom}_{event['from_allele']}" #parent allele
            dup_chrom_key:str = f"{chrom}_{event['to_allele']}" #new allele
            genome[dup_chrom_key] = genome[chrom_key]

            ## Update position modifiers for the new allele
            ### Germline
            tmp_germ:pd.DataFrame = germ_info[(germ_info['chrom'].astype(str) == chrom) & (germ_info['allele'] == event['from_allele'])]
            tmp_germ.loc[:,'allele'] = event['to_allele']
            germ_info = pd.concat([germ_info, tmp_germ]).reset_index(drop=True)
            ### Somatic
            tmp_som:pd.DataFrame = somatic_info[(somatic_info['chrom'].astype(str) == chrom) & (somatic_info['allele'] == event['from_allele'])]
            tmp_som.loc[:,'allele'] = event['to_allele']
            somatic_info = pd.concat([somatic_info, tmp_som]).reset_index(drop=True)

    # Introduce the CNA into the reference genome
    sv_info:pd.DataFrame = pd.DataFrame(columns=list(germ_info.columns))
    for _,sv in sv_df.iterrows():
        chrom:str = str(sv['chrom1'])
        start:int = sv['start1']
        end:int = sv['end2']
        sv_event:pd.Series = events_df[events_df['event_id'] == sv['sv_id']].iloc[0]
        parent_allele:str = f'{chrom}_allele_2_major' if 'major' in sv_event['from_allele'] else f'{chrom}_allele_1_minor'

        ## Get coordinates
        parent_start, parent_end, parent_mov = get_sv_mov(germ_info, somatic_info, sv_info, chrom, start, end, parent_allele)
        event_start, event_end, event_mov = get_sv_mov(germ_info, somatic_info, sv_info, chrom, start, end, event['to_allele'])

        if sv['svclass'] == "DUP":
            genome[parent_allele] = genome[parent_allele][:parent_end] + genome[f"{chrom}_{sv_event['to_allele']}"][event_start:event_end] + genome[parent_allele][parent_end:]
            parent_mov += end-start
            sv_info = pd.concat([sv_info, pd.DataFrame(data={'chrom': [chrom], 'pos': [end], 'allele': [parent_allele], 'mov': [parent_mov]})])
        elif sv['svclass'] == "DEL":
            genome[parent_allele] = genome[parent_allele][:parent_start] + genome[parent_allele][parent_end:]
            parent_mov -= end-start
            sv_info = pd.concat([sv_info, pd.DataFrame(data={'chrom': [chrom], 'pos': [end], 'allele': [parent_allele], 'mov': [parent_mov]})])

    # Write the new genome
    whole_genome:str = ''
    for chrom in genome.keys():
        if ('allele_2_major' in chrom) or ('allele_1_minor' in chrom):
            whole_genome += genome[chrom] + 'N'*1000
    custom_genome_path:click.Path = os.path.join(outDir, f"{donor_id}_tmp", f"{donor_id}_genome.fa")
    with open(custom_genome_path, 'w') as custom_genome:
        custom_genome.write(f'>custom_genome\n{whole_genome}\n')

def insilicoseq_docker(cpus:int, refGenome:click.Path, cov:int, model:str, outDir:click.Path, donor_id:str) -> None:

    """
    Run InSilicoSeq independently for each chromosome
    """

    custom_genome_path:click.Path = os.path.join(outDir, f"{donor_id}_tmp", f"{donor_id}_genome.fa")

    # Calculate number of reads to simulate based on the coverage and the length of the genome
    chromosomes = pd.read_csv(f'{refGenome}.fai', delimiter="\t", usecols=[0,1], names=['chrom', 'length'])
    total_len:int = chromosomes['length'].sum()
    model_read_length:dict = {"HiSeq":125,"NextSeq":300,"NovaSeq":150,"MiSeq":300}
    n_reads:int = round((cov*total_len)/model_read_length[model])

    # Command
    cmd:list = [
        "iss", "generate",
        "--cpus", f"{cpus}",
        "--genomes", custom_genome_path,
        "--output", os.path.join(outDir, f"{donor_id}_fastq", f"{donor_id}_reads"),
        "--n_reads", str(n_reads),
        "--model", f"{model}",
        "--gc_bias", "--compress", "--store_mutations"]
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

    # Extract error mutations added by InSilicoSeq
    os.rename(os.path.join(outDir, f"{donor_id}_fastq", f"{donor_id}_reads.vcf.gz"), os.path.join(outDir, f"{donor_id}_cna_error_muts.vcf.gz"))
    
    # Remove tmp files
    os.remove(custom_genome_path)
    os.remove(os.path.join(outDir, f"{donor_id}_fastq", f"{donor_id}_reads_abundance.txt"))

@click.command(name="InSilicoSeq-CNA")
@click.option("-@", "--cpus",
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of CPUs to use for read generation")
@click.option("-i", "--input",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="VCF file containing the mutations simulated by OncoGAN")
@click.option("--events",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="TSV file containing the order of the events simulated by OncoGAN")
@click.option("--sv",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="TSV file containing the structural variants simulated by OncoGAN")
@click.option("-o", "--outDir", "outDir",
              type=click.Path(exists=False, file_okay=False),
              default=os.getcwd(),
              show_default=False,
              help="Directory where save the BAMs. Default is the current directory")
@click.option("-r", "--refGenome", "refGenome",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="Reference genome in fasta format")
@click.option("--dbSNP", "dbSNP",
              type=click.Path(exists=True, file_okay=True),
              required=False,
              default=None,
              help="VCF file containing the germline mutations to be added to the reference genome")
@click.option("-c", "--coverage",
              type=click.INT,
              default=30,
              show_default=True,
              help="Genome coverage to simulate")
@click.option("-m", "--model",
              type=click.Choice(["HiSeq", "NextSeq", "NovaSeq", "MiSeq", "MiSeq-20", "MiSeq-24", "MiSeq-28", "MiSeq-32"]),
              metavar="TEXT",
              show_choices=False,
              default="NovaSeq",
              show_default=True,
              help="Use HiSeq (125bp/read), NextSeq(300bp/read), NovaSeq(125bp/read), MiSeq (300bp/read) or MiSeq-[20,24,28,32](300bp/read) for a pre-computed error model provided with InSilicoSeq (v2.0.1)")
def InSilicoSeq_CNA(cpus, input, events, sv, outDir, refGenome, dbSNP, coverage, model):

    """
    Wrapper to run InSilicoSeq (v2.0.1) with OncoGAN-CNA simulations
    """

    # Load the VCF
    vcf:pd.DataFrame = readVCF(input)
    donor_id:str = vcf['id'][0]

    # Create directories
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    os.makedirs(os.path.join(outDir, f"{donor_id}_tmp"))
    os.makedirs(os.path.join(outDir, f"{donor_id}_fastq"))
    
    # Load reference genome
    genome:Fasta = Fasta(refGenome)

    # dbSNP
    if dbSNP is not None:
        ## Load dbSNP
        dbsnp_vcf:pd.DataFrame = readVCF(dbSNP, dbSNP=True)

        ## Add dbSNP polymorphisms
        genome, dbsnp_vcf, updated_positions = introduce_polymorphisms(genome, dbsnp_vcf)
        dbsnp_vcf.to_csv(os.path.join(outDir, os.path.basename(dbSNP).replace('.vcf', '_with_alleles.tsv')), sep='\t', index=False)
    else:
        genome, updated_positions = initialize_genome(genome)

    # Add mutations
    introduce_mutations(genome, vcf, updated_positions, events, sv, outDir, donor_id)

    # Run InsilicoSeq 
    insilicoseq_docker(cpus, refGenome, coverage, model, outDir, donor_id)

    # Remove temporal directory
    os.removedirs(os.path.join(outDir, f"{donor_id}_tmp"))

    # Concatenate fastqs
    ## R1
    r1_files:list = sorted(glob.glob(os.path.join(outDir, f"{donor_id}_fastq", "*R1.fastq.gz")))
    with open(os.path.join(outDir, f"{donor_id}_cna_R1.fastq.gz"), 'w') as out_file:
        subprocess.run(["cat"] + r1_files, stdout=out_file, check=True)
    
    ## R2
    r2_files:list = sorted(glob.glob(os.path.join(outDir, f"{donor_id}_fastq", "*R2.fastq.gz")))
    with open(os.path.join(outDir, f"{donor_id}_cna_R2.fastq.gz"), 'w') as out_file:
        subprocess.run(["cat"] + r2_files, stdout=out_file, check=True)
    
    ## Remove individual fastqs
    subprocess.run(["rm", "-rf", os.path.join(outDir, f"{donor_id}_fastq")], check=True)

if __name__ == '__main__':
    InSilicoSeq_CNA()
