#!/usr/local/bin/python3

# Import model path
import sys
sys.path.append('/genomeGAN/ctabgan/')

# Import modules
import os
import click
import pandas as pd
from model.ctabgan import CTABGAN
from model.eval.evaluation import get_utility_metrics,stat_sim,privacy_metrics
import torch

torch.cuda.empty_cache()


# CLI options
@click.command(name='simulateTrainedCounts')
@click.option("--pkl",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="CTABGAN model file for counts")
@click.option('--nFiles', 'nFiles',
              type=click.INT,
              required=False,
              default = 1,
              help="Number of files to generate")
@click.option('--nSamples', 'nSamples',
              type=click.INT,
              required=False,
              default = 50,
              help="Number of donors to simulate for each file")
@click.option('--outdir',
              type=click.Path(exists=True, file_okay=False),
              default=os.getcwd(),
              show_default=True,
              help="Directory where save the model")
def simulateTrainedCounts(pkl, nFiles, nSamples, outdir) -> None:

    """
    Simulate some counts files from trained models for QC purposes
    """

    # Get output name
    outname:str = os.path.basename(pkl).split(".")[0]

    # Torch options
    device:str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model
    countModel = torch.load(pkl, map_location=device)

    # Generate the simulations
    for n in range(nFiles):
        syn:pd.DataFrame = countModel.generate_samples(nSamples)
        syn.to_csv(f"{outdir}/sim{n}_{outname}.txt", sep="\t", index=False)

if __name__ == '__main__':
    simulateTrainedCounts()