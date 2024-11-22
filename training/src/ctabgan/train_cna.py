#!/usr/local/bin/python3

# Import model path
import sys
sys.path.append('/oncoGAN/ctabgan/')

# Import modules
import os
import click
import pandas as pd
from model.ctabgan import CTABGAN
from model.eval.evaluation import get_utility_metrics,stat_sim,privacy_metrics
from torch import cuda, save

cuda.empty_cache()

# CLI options
@click.command(name='trainCNA')
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
@click.option("--categorical_columns",
              type=click.STRING,
              default = None,
              help="Categorical columns. Comma separated with no space (e.g. x,y,z)")
@click.option("--log_columns",
              type=click.STRING,
              default = None,
              help="Log columns. Comma separated with no space (e.g. x,y,z)")
@click.option("--integer_columns",
              type=click.STRING,
              default = None,
              help="Integer columns. Comma separated with no space (e.g. x,y,z)")
def trainCNA(csv, prefix, outdir, epochs, batch_size, test_ratio, lr, categorical_columns, log_columns, integer_columns):

    """
    Train a CTABGAN model for artifacts
    """
      
    # Parse columns type
    categorical_columns = [] if categorical_columns == None else categorical_columns.strip().split(',')
    log_columns = [] if log_columns == None else log_columns.strip().split(',')
    integer_columns = [] if integer_columns == None else integer_columns.strip().split(',')

    # Get training file information
    n:int = pd.read_csv(csv).shape[0]

    # Initializing the synthesizer object and specifying input parameters
    ## Notice: If you have continuous variable, you do not need to explicitly assign it. It will be treated like that by default
    synthesizer = CTABGAN(raw_csv_path = csv,
                          test_ratio = test_ratio,
                          categorical_columns = categorical_columns,
                          log_columns = log_columns,
                          mixed_columns = {},
                          general_columns= [],
                          integer_columns = integer_columns,
                          problem_type = {None: None},
                          epochs = epochs,
                          batch_size = batch_size,
                          lr = lr)
    
    # Fitting the synthesizer to the training dataset
    synthesizer.fit()

    # Generating synthetic data as test
    syn = synthesizer.generate_samples(n)
    if len(syn) < n:
        print(f'Error during sample generation for {prefix}: epoch={epochs} batchsize={batch_size} lr={lr}')
    else:
        save(synthesizer, f"{outdir}/{prefix}_CNA_epoch{epochs}_batchsize{batch_size}_lr{lr}_testratio{test_ratio}.pkl")
        syn.to_csv(f"{outdir}/{prefix}_CNA_epoch{epochs}_batchsize{batch_size}_lr{lr}_testratio{test_ratio}.txt", sep="\t", index=False)

if __name__ == '__main__':
    trainCNA()