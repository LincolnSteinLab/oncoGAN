import os
import glob
import click
import subprocess
import random
import itertools
import pandas as pd
import numpy as np
from pyfaidx import Fasta, Sequence

def readVCF(input:click.Path) -> pd.DataFrame:

    """
    A function to open the VCF as a pd DataFrame
    """
    
    vcf:pd.DataFrame = pd.read_table(input, sep = "\t", comment = '#',
                        names = ['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'])
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

def assign_copies(row:pd.Series, combinations:dict, copies:list) -> str:
    
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
def InSilicoSeq(cpus, input, outDir, refGenome, coverage, model):

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

    # Work on each chromosome individually
    for chrom in genome.keys():
        ## Variables
        tmp_genome:dict = {}
        final_genome_path:click.Path = os.path.join(outDir, f"{donor_id}_tmp", f"{donor_id}_genome{chrom}.fa")
        abundance_path:click.Path = os.path.join(outDir, f"{donor_id}_tmp", f"{donor_id}_abundance{chrom}.txt")
        
        ## Create a temporal genome and abundance file for each chromosome
        with open(abundance_path, 'w') as tmp_abundance:
            for freq in ['0.3', '0.15', '0.05']:
                for hap in ['A', 'B']:
                    tmp_abundance.write(f'{chrom}_freq{freq}_hap{hap}\t{freq}\n')
                    tmp_genome[f'{chrom}_freq{freq}_hap{hap}'] = genome[chrom][:]

        
        ## Filter the VCF
        vcf['chrom'] = vcf['chrom'].astype(str)
        vcf_subset:pd.DataFrame = vcf[vcf['chrom'] == str(chrom)]
        if vcf_subset.empty:
            with open(final_genome_path, 'w') as final_genome: 
                for copy_chrom in tmp_genome:
                    final_genome.write(f'>{copy_chrom}\n{tmp_genome[copy_chrom]}\n')
            continue
        
        ## Randomly assign haplotypes for each mutation
        vcf_subset['hap'] = vcf_subset['af'].apply(lambda af: random.choice(['AB', 'BA']) if af > 0.6 else random.choice(['A', 'B']))

        ## Assign chromosome copies
        copies:list = [0.3, 0.15, 0.05]
        possible_combinations:dict = {}
        for r in range(1, len(copies) + 1):
            for subset in itertools.combinations(copies, r):
                s:float = sum(subset)
                if s not in possible_combinations:
                    possible_combinations[s] = subset
        vcf_subset['copy_assignment'] = vcf_subset.apply(assign_copies, combinations=possible_combinations, copies=copies, axis=1)

        ## Create the genome with the mutations
        with open(final_genome_path, 'w') as final_genome:
            for copy_chrom in tmp_genome:
                tmp_vcf:pd.DataFrame = vcf_subset[vcf_subset['copy_assignment'].str.contains(copy_chrom)]
                if tmp_vcf.empty:
                    final_genome.write(f'>{copy_chrom}\n{tmp_genome[copy_chrom]}\n')
                    continue
                else:
                    mov = 0
                    error_muts:pd.DataFrame = pd.DataFrame()
                    new_allele:Sequence = str(tmp_genome[copy_chrom])
                    for _,mut in tmp_vcf.iterrows():
                        position = mut['pos']+mov-1
                        if ((len(mut['ref']) == 1) and (len(mut['alt']) == 1)): #SNP
                            if mut['ref'][0] != new_allele[position]:
                                error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                                continue
                            else:
                                new_allele = new_allele[:position] + mut['alt'] + new_allele[position+1:]
                                continue
                        elif ((len(mut['ref']) == 2) and (len(mut['alt']) == 2)): #DNP
                            if mut['ref'] != new_allele[position:position+2]:
                                error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                                continue
                            else:
                                new_allele = new_allele[:position] + mut['alt'] + new_allele[position+2:]
                                continue
                        elif ((len(mut['ref']) == 3) and (len(mut['alt']) == 3)): #TNP
                            if mut['ref'][0] != new_allele[position:position+3]:
                                error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                                continue
                            else:
                                new_allele = new_allele[:position] + mut['alt'] + new_allele[position+3:]
                                continue
                        elif (len(mut['ref']) > 1): #DEL
                            if mut['ref'][0] != new_allele[position]:
                                error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                                continue
                            else:
                                new_allele = new_allele[:position+1] + new_allele[position+len(mut['ref']):]
                                mov -= len(mut['ref'])-1
                                continue
                        elif (len(mut['alt']) > 1): #INS
                            if mut['ref'][0] != new_allele[position]:
                                error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                                continue
                            else:
                                new_allele = new_allele[:position] + mut['alt'] + new_allele[position+1:]
                                mov += len(mut['alt'])-1
                                continue
                    final_genome.write(f'>{copy_chrom}\n{new_allele}\n')
        
        # Run InsilicoSeq independently for each chromosome
        ## Calculate number of reads
        model_read_length:dict = {"HiSeq":125,"NextSeq":300,"NovaSeq":150,"MiSeq":300}
        n_reads:int = round((coverage*len(genome[chrom]))/model_read_length[model])

        ## Command
        cmd:list = [
            "docker", "run", "--rm",
            "-u", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{os.path.abspath(outDir)}:/mnt/data",
            "-it", "quay.io/biocontainers/insilicoseq:2.0.1--pyh7cba7a3_0",
            "iss", "generate",
            "--cpus", f"{cpus}",
            "--genomes", f"/mnt/data/{donor_id}_tmp/{donor_id}_genome{chrom}.fa",
            "--abundance_file", f"/mnt/data/{donor_id}_tmp/{donor_id}_abundance{chrom}.txt",
            "--output", f"/mnt/data/{donor_id}_fastq_by_chrom/{donor_id}_reads_{chrom}",
            "--n_reads", f"{n_reads}",
            "--model", f"{model}",
            "--gc_bias", "--compress", "--store_mutations"]
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

        ## Extract error mutations added by InSilicoSeq
        os.rename(os.path.join(outDir, f"{donor_id}_fastq_by_chrom", f"{donor_id}_reads_1.vcf.gz"), os.path.join(outDir, f"{donor_id}_error_muts.vcf.gz"))
        
        ## Remove tmo files
        os.remove(final_genome_path)
        os.remove(abundance_path)

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
    