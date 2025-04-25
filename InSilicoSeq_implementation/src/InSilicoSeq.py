import os
import glob
import click
import subprocess
import random
import itertools
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
        vcf[['af', 'ms']] = vcf['info'].str.extract(r'AF=([\d.]+);MS=([A-Za-z0-9]+)')
        vcf['af'] = vcf['af'].astype(float)
        vcf = vcf.drop(columns=['qual', 'filter', 'info', 'ms'])
    
    return vcf

def best_combination(combinations:dict, af:float) -> tuple:
    
    """
    Return the subset of copies whose sum is closest to the target allele frequency
    """
    
    best_key:int = min(combinations.keys(), key=lambda k: abs(k - af))
    return combinations[best_key]

def assign_copies_apply(row:pd.Series, combinations:dict, copies:list) -> str:
    
    """
    Assign to which chromosome copies each mutation should be introduced based on its allele frequency
    """

    chrom:str = row['chrom']
    af:float = row['af']
    hap:str = row['hap']
    
    if af < 0.6:
        comb:tuple = best_combination(combinations, af)
        assignment_str:str = ','.join([f"{chrom}_freq{freq}_hap{hap}" for freq in comb])
        return assignment_str
    else:
        ## Major haplotype
        major_assignment:list = [f"{chrom}_freq{freq}_hap{hap[0]}" for freq in copies]
        ## Minor haplotype
        comb_minor:tuple = best_combination(combinations, af-sum(copies))
        minor_assignment:list = [f"{chrom}_freq{freq}_hap{hap[1]}" for freq in comb_minor]
        assignment:list = major_assignment + minor_assignment
        assignment_str:str = ','.join(assignment)
        return assignment_str

def introduce_polymorphisms(genome:Fasta, dbsnp:pd.DataFrame) -> tuple:

    """
    Add polymorphisms to the reference genome
    """

    # Randomly assign haplotypes for each SNP
    dbsnp['allele'] = random.choices(['A', 'B', 'AB'], k = dbsnp.shape[0])

    updated_genome:dict = {}
    updated_positions:pd.DataFrame = pd.DataFrame()
    for chrom in genome.keys():
        for allele in ['A', 'B']:
            mov:int = 0
            error_muts:pd.DataFrame = pd.DataFrame() 
            updated_chrom:str = str(genome[chrom])

            chrom_dbsnp:pd.DataFrame = dbsnp[(dbsnp['chrom'].astype('str') == str(chrom)) & (dbsnp['allele'].isin([allele, 'AB']))]
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

def assign_allele_copies(vcf:pd.DataFrame) -> pd.DataFrame:

    """
    Assign to which chromosome copies each mutation should be introduced based on its allele frequency
    """

    vcf['hap'] = vcf['af'].apply(lambda af: random.choice(['AB', 'BA']) if af > 0.6 else random.choice(['A', 'B']))
    copies:list = [0.3, 0.15, 0.05]
    possible_combinations:dict = {}
    for r in range(1, len(copies) + 1):
        for subset in itertools.combinations(copies, r):
            s:float = sum(subset)
            if s not in possible_combinations:
                possible_combinations[s] = subset
    vcf['copy_assignment'] = vcf.apply(assign_copies_apply, combinations=possible_combinations, copies=copies, axis=1)

    return vcf

def subset_mutations_apply(x:pd.Series, key:str) -> bool:

    """
    Filter mutations by their assigned chromosome/allele/frequency
    """

    x = x.split(',')
    if key in x:
        return True
    else:
        return False

def get_germline_mov(germ_pos:pd.DataFrame, pos:int) -> int:

    """
    Keep the trace of the length difference of the custom genome with respect to the reference genome
    """

    germ_pos_filtered:pd.DataFrame = germ_pos[germ_pos['pos'] < pos]
    if germ_pos_filtered.empty:
        return 0
    else:
        last_row:pd.Series = germ_pos_filtered.tail(1).iloc[0]
        return last_row['mov']

def introduce_mutations(genome:dict, mutations:pd.DataFrame, germ_info:pd.DataFrame, outDir:click.Path, donor_id:str) -> pd.DataFrame:

    """
    Add mutations to the custom reference genome
    """

    # Randomly assign haplotypes for each mutation
    mutations = assign_allele_copies(mutations)

    chrom_lengths:dict = {}
    for chrom_allele in genome.keys():
        chrom:str = str(chrom_allele.split('_')[0])
        allele:str = chrom_allele.split('_')[1]
        chrom_germ_info:pd.DataFrame = germ_info[(germ_info['chrom'].astype(str) == chrom) & (germ_info['allele'] == allele)]


        final_genome_path:click.Path = os.path.join(outDir, f"{donor_id}_tmp", f"{donor_id}_genome{chrom}.fa")
        abundance_path:click.Path = os.path.join(outDir, f"{donor_id}_tmp", f"{donor_id}_abundance{chrom}.txt")
        with open(final_genome_path, 'a') as final_genome, open(abundance_path, 'a') as abundance: 
            for freq in ['0.3', '0.15', '0.05']:
                chrom_freq_allele = f'{chrom}_freq{freq}_hap{allele}'

                ## Create an abundance file for each chromosome to use with InSilicoSeq
                abundance.write(f'{chrom_freq_allele}\t{freq}\n')
                
                ## Subset mutations
                chrom_mutations:pd.DataFrame = mutations[(mutations['chrom'].astype('str') == str(chrom))]
                chrom_mutations['keep'] = mutations['copy_assignment'].apply(lambda x: subset_mutations_apply(x, key=chrom_freq_allele))
                chrom_mutations = chrom_mutations[chrom_mutations['keep']]

                if chrom_mutations.empty:
                    final_genome.write(f'>{chrom_freq_allele}\n{genome[chrom_allele]}\n')
                    continue

                mov:int = 0
                error_muts:pd.DataFrame = pd.DataFrame()
                updated_chrom:str = genome[chrom_allele]
                for _,mut in chrom_mutations.iterrows():
                    chrom_germ_mov:int = get_germline_mov(chrom_germ_info, mut['pos'])
                    position = mut['pos']+mov+chrom_germ_mov-1
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
                            continue
                    elif (len(mut['alt']) > 1): #INS
                        if mut['ref'][0] != updated_chrom[position]:
                            error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                            continue
                        else:
                            updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+1:]
                            mov += len(mut['alt'])-1
                            continue

                final_genome.write(f'>{chrom_freq_allele}\n{updated_chrom}\n')
                chrom_lengths[chrom_freq_allele] = len(updated_chrom)
    return(mutations, chrom_lengths)

def insilicoseq_docker(cpus:int, refGenome:click.Path, cov:int, model:str, outDir:click.Path, donor_id:str) -> None:

    """
    Run InSilicoSeq independently for each chromosome
    """

    chromosomes = pd.read_csv(f'{refGenome}.fai', delimiter="\t", usecols=[0,1], names=['chrom', 'length'])
    for _,row in chromosomes.iterrows():
        chrom = str(row['chrom'])
        chrom_fasta:click.Path = os.path.join(outDir, f"{donor_id}_tmp", f"{donor_id}_genome{chrom}.fa")
        chrom_abundance:click.Path = os.path.join(outDir, f"{donor_id}_tmp", f"{donor_id}_abundance{chrom}.txt")

        # Calculate number of reads
        model_read_length:dict = {"HiSeq":125,"NextSeq":300,"NovaSeq":150,"MiSeq":300}
        n_reads:int = round((cov*int(row['length']))/model_read_length[model])

        # Check if the genome exists
        if not os.path.exists(chrom_fasta):
            continue

        # Command
        cmd:list = [
            "iss", "generate",
            "--cpus", f"{cpus}",
            "--genomes", chrom_fasta,
            "--abundance_file", chrom_abundance,
            "--output", os.path.join(outDir, f"{donor_id}_fastq_by_chrom", f"{donor_id}_reads_{chrom}"),
            "--n_reads", f"{n_reads}",
            "--model", f"{model}",
            "--gc_bias", "--compress", "--store_mutations"]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

        # Extract error mutations added by InSilicoSeq
        os.rename(os.path.join(outDir, f"{donor_id}_fastq_by_chrom", f"{donor_id}_reads_1.vcf.gz"), os.path.join(outDir, f"{donor_id}_error_muts.vcf.gz"))
        
        # Remove tmp files
        os.remove(chrom_fasta)
        os.remove(chrom_abundance)

@click.command(name="InSilicoSeq")
@click.option("-@", "--cpus",
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of CPUs to use for read generation")
@click.option("-i", "--input",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="VCF file containing the desired mutations")
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
              required=True,
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
def InSilicoSeq(cpus, input, outDir, refGenome, dbSNP, coverage, model):

    """
    Wrapper to run InSilicoSeq (v2.0.1) with OncoGAN simulations
    """

    # Load the VCF
    vcf:pd.DataFrame = readVCF(input)
    donor_id:str = vcf['id'][0]

    # Create directories
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    os.makedirs(os.path.join(outDir, f"{donor_id}_tmp"))
    os.makedirs(os.path.join(outDir, f"{donor_id}_fastq_by_chrom"))
    
    # Load reference genome
    genome:Fasta = Fasta(refGenome)

    # Load dbSNP
    dbsnp_vcf:pd.DataFrame = readVCF(dbSNP, dbSNP=True)

    # Add dbSNP polymorphisms
    genome, dbsnp_vcf, updated_positions = introduce_polymorphisms(genome, dbsnp_vcf)
    dbsnp_vcf.to_csv(os.path.join(outDir, os.path.basename(dbSNP).replace('.vcf', '_with_alleles.tsv')), sep='\t', index=False)

    # Add mutations
    vcf_assigned, chrom_lengths = introduce_mutations(genome, vcf, updated_positions, outDir, donor_id)
    vcf_assigned.to_csv(os.path.join(outDir, os.path.basename(input).replace('.vcf', '_with_alleles.tsv')), sep='\t', index=False)

    # Run InsilicoSeq 
    insilicoseq_docker(cpus, refGenome, coverage, model, outDir, donor_id)

    # Remove temporal directory
    os.removedirs(os.path.join(outDir, f"{donor_id}_tmp"))

    # Concatenate fastqs
    ## R1
    r1_files:list = sorted(glob.glob(os.path.join(outDir, f"{donor_id}_fastq_by_chrom", "*R1.fastq.gz")))
    with open(os.path.join(outDir, f"{donor_id}_R1.fastq.gz"), 'w') as out_file:
        subprocess.run(["cat"] + r1_files, stdout=out_file, check=True)
    
    ## R2
    r2_files:list = sorted(glob.glob(os.path.join(outDir, f"{donor_id}_fastq_by_chrom", "*R2.fastq.gz")))
    with open(os.path.join(outDir, f"{donor_id}_R2.fastq.gz"), 'w') as out_file:
        subprocess.run(["cat"] + r2_files, stdout=out_file, check=True)
    
    ## Remove individual fastqs
    subprocess.run(["rm", "-rf", os.path.join(outDir, f"{donor_id}_fastq_by_chrom")], check=True)

if __name__ == '__main__':
    InSilicoSeq()
    