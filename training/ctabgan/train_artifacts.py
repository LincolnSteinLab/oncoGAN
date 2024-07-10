#!/usr/local/bin/python3

# Import model path
import sys
sys.path.append('/genomeGAN/ctabgan/')

# Import modules
import os
import click
from model.ctabgan import CTABGAN
from model.eval.evaluation import get_utility_metrics,stat_sim,privacy_metrics
from torch import cuda, save

cuda.empty_cache()

# CLI options
@click.command(name='trainArtifacts')
@click.option("--csv",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="CSV file with the artifacts used to train the model")
@click.option('--prefix',
              type=click.STRING,
              required=True,
              help="Prefix to use to save the model")
@click.option('--outdir',
              type=click.Path(exists=True, file_okay=False),
              default=os.getcwd(),
              show_default=True,
              help="Directory where save the model")
@click.option("--epochs",
              type=click.INT,
              default=800,
              show_default=True,
              help="Number of epochs")
@click.option("--batch_size",
              type=click.INT,
              default=2000,
              show_default=True,
              help="Batch size")
@click.option("--test_ratio",
              type=click.FLOAT,
              default=0.3,
              show_default=True,
              help="Test ratio")
@click.option("--lr",
              type=click.FLOAT,
              default=2e-4,
              show_default=True,
              help="Learning rate")
def trainArtifacts(csv, prefix, outdir, epochs, batch_size, test_ratio, lr):

    """
    Train a CTABGAN model for artifacts
    """
      
    # Initializing the synthesizer object and specifying input parameters
    ## Notice: If you have continuous variable, you do not need to explicitly assign it. It will be treated like that by default
    synthesizer = CTABGAN(raw_csv_path = csv,
                          test_ratio = test_ratio,
                          categorical_columns = ['ctx1', 'ctx2', 'ctx3', 'ctx4', 'ref', 'alt', 'artifact'],
                          log_columns = [],
                          mixed_columns = {},
                          general_columns= [],
                          integer_columns = ['start'],
                          problem_type= {"Classification": "artifact"},
                          epochs = epochs,
                          batch_size = batch_size,
                          lr = lr)
    
    # Fitting the synthesizer to the training dataset
    synthesizer.fit()

    # Generating synthetic data as test
    syn = synthesizer.generate_samples(100)
    if len(syn) < 100:
        print(f'Error during sample generation for {prefix}: epoch={epochs} batchsize={batch_size} lr={lr}')
    else:
        save(synthesizer, f"{outdir}/{prefix}_artifacts_epoch{epochs}_batchsize{batch_size}_lr{lr}_testratio{test_ratio}.pkl")
        syn.to_csv(f"{outdir}/{prefix}_artifacts_epoch{epochs}_batchsize{batch_size}_lr{lr}_testratio{test_ratio}.txt", sep="\t", index=False)

if __name__ == '__main__':
    trainArtifacts()