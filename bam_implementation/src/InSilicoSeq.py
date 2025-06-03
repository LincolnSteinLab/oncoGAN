import os
import click
import subprocess
import pandas as pd

def calculate_n_reads(ref_index:str, coverage:int, model:str) -> int:

    """
    Calculate the number of reads to simulate based on the coverage and the length of the genome
    """
    
    # Read the reference genome index file
    chromosomes:pd.DataFrame = pd.read_csv(ref_index, delimiter="\t", usecols=[0,1], names=['chrom', 'length'])
    total_len:int = chromosomes['length'].sum()
    model_read_length:dict = {"HiSeq":125,"NextSeq":300,"NovaSeq":150,"MiSeq":300}
    n_reads:int = round((coverage*total_len)/model_read_length[model])

    return n_reads

@click.command(name="InSilicoSeq")
@click.option("-@", "--cpus",
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of CPUs to use for read generation")
@click.option("-p", "--prefix",
              type=click.STRING,
              required=True,
              help="FASTQ files prefix")
@click.option("-g", "--custom_genome",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="Custom genome in fasta format")
@click.option("-r", "--reference_genome",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="Reference genome used to generate the custom genome. Required to calculate the number of reads to simulate")
@click.option("-a", "--abundance",
              type=click.Path(exists=True, file_okay=True),
              required=False,
              default=None,
              help="Abundance file to guide the simulation [optional]")
@click.option("-o", "--out_dir",
              type=click.Path(exists=False, file_okay=False),
              default=os.getcwd(),
              show_default=False,
              help="Directory where save the BAMs. Default is the current directory")
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
def InSilicoSeq(cpus, prefix, custom_genome, reference_genome, abundance, out_dir, coverage, model):

    """
    Run InSilicoSeq
    """

    # Check if the genome is indexed
    reference_index:click.Path = f"{reference_genome}.fai"
    if not os.path.exists(reference_index):
        print(f"Indexing {os.path.basename(reference_genome)} with samtools faidx")
        cmd:list = ["samtools", "faidx", reference_genome]
        subprocess.run(cmd, check=True)

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Calculate number of reads to simulate based on the coverage and the length of the genome
    n_reads:int = calculate_n_reads(reference_index, coverage, model)

    # Command
    cmd:list = [
        "iss", "generate",
        "--cpus", f"{cpus}",
        "--genomes", custom_genome,
        "--output", os.path.join(out_dir, prefix),
        "--n_reads", str(n_reads),
        "--model", f"{model}",
        "--gc_bias", "--compress", "--store_mutations"
    ]
    if abundance is not None:
        cmd.extend(["--abundance_file", abundance])

    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
    
    # Rename error mutations added by InSilicoSeq
    os.rename(os.path.join(out_dir, f"{prefix}.vcf.gz"), os.path.join(out_dir, f"{prefix}_error_muts.vcf.gz"))
