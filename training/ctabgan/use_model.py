#!/usr/local/bin/python3

# Import model path
import sys
sys.path.append('/genomeGAN/')

# Import modules
import os
import click
from torch import load
import pandas as pd

# CLI options
@click.command(name='useModel')
@click.option("--model",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="Model to use for the simulations (counts/drivers)")
@click.option('--outdir',
              type=click.Path(exists=True, file_okay=False),
              default=os.getcwd(),
              show_default=True,
              help="Directory where save the simulations")
@click.option('--nDonors', 'nDonors',
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of donors to simulate for each simulation")
@click.option('--nFiles', 'nFiles',
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of files to simulate")
def useModel(model, outdir, nDonors, nSimulations):

    """
    Use a driver/count CTABGAN model to generate synthetic data
    """

    # Open the model
    synthesizer = load(model, map_location="cpu")

    # Simulate the desire number of files
    for i in range(nSimulations):
        syn:pd.DataFrame = synthesizer.generate_samples(nDonors)
        modelName:str = os.path.basename(model).strip('.pkl')
        syn.round(0).astype(int).to_csv(f"{outdir}/sim{i}_{modelName}.txt", sep="\t", index=False)
    
if __name__ == '__main__':
    useModel()