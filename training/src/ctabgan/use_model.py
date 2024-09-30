#!/usr/local/bin/python3

# Import model path
import sys
sys.path.append('/oncoGAN/')

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
              help="Model to use for the simulations")
@click.option("--feature",
              type=click.Choice(['counts','drivers', 'artifacts', 'cna','sv'], case_sensitive=False),
              required=True,
              help="Feature to simulate")
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
def useModel(model, feature, outdir, nDonors, nFiles):

    """
    Use a CTABGAN model to generate synthetic data
    """

    # Open the model
    synthesizer = load(model, map_location="cpu")

    # Simulate the desire number of files
    for i in range(nFiles):
        syn:pd.DataFrame = synthesizer.generate_samples(nDonors)
        modelName:str = os.path.basename(model).strip('.pkl')
        if feature == "counts" or feature == "drivers":
            syn.round(0).astype(int).to_csv(f"{outdir}/sim{i}_{modelName}.txt", sep="\t", index=False)
        else:
            syn.to_csv(f"{outdir}/sim{i}_{modelName}.txt", sep="\t", index=False)

if __name__ == '__main__':
    useModel()