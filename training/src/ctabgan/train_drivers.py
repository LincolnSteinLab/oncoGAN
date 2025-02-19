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

def trainDrivers(csv, prefix, outdir, epochs, batch_size, lr, test_ratio, categorical_columns, log_columns, integer_columns, mixed_columns, general_columns, tqdm_disable) -> None:

    # Get training file information
    colnames:list = pd.read_csv(csv, nrows=1).columns.tolist()
    n:int = pd.read_csv(csv).shape[0]

    # Initializing the synthesizer object and specifying input parameters
    ## Notice: If you have continuous variable, you do not need to explicitly assign it. It will be treated like that by default
    synthesizer = CTABGAN(raw_csv_path = csv,
                          test_ratio = test_ratio,
                          categorical_columns = categorical_columns,
                          log_columns = log_columns,
                          mixed_columns = mixed_columns,
                          general_columns= general_columns,
                          integer_columns = integer_columns,
                          problem_type= {None: None},
                          epochs = epochs,
                          batch_size = batch_size,
                          lr = lr,
                          tqdm_disable = tqdm_disable)
    
    # Fitting the synthesizer to the training dataset
    tries:int = 0
    max_tries:int = 7
    while tries <= max_tries:
        synthesizer.fit()
        
        # Generating synthetic data as test
        syn:pd.DataFrame = synthesizer.generate_samples(n)
        if len(syn) < n:
            if tries == max_tries:
                print(f'Error during sample generation for {prefix}: epoch={epochs} batchsize={batch_size} lr={lr} in {max_tries} tries')
                break
            else:
                tries += 1
        else:
            save(synthesizer, f"{outdir}/{prefix}_drivers_epoch{epochs}_batchsize{batch_size}_lr{lr}.pkl")
            syn.round(0).astype(int).to_csv(f"{outdir}/{prefix}_drivers_epoch{epochs}_batchsize{batch_size}_lr{lr}.txt", sep="\t", index=False)
            break

# CLI options
@click.command(name='trainDrivers')
@click.option("--csv",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="CSV file with the counts used to train the model")
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
              default=100,
              show_default=True,
              help="Number of epochs")
@click.option("--batch_size",
              type=click.INT,
              default=20,
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
@click.option("--mixed_columns",
              type=click.STRING,
              default = None,
              help="mixed columns. Dictionary string with the variable name as the key and integer values that should be considered categorical in a list format (e.g. \"{'len': [0]}\")")
@click.option("--general_columns",
              type=click.STRING,
              default = None,
              help="general columns. Comma separated with no space (e.g. x,y,z)")
@click.option('--no-tqdm', 'tqdm_disable',
              is_flag=True,
              flag_value=True,
              required=False,
              help="Disable tqdm progress bar")
def trainDriversClick(csv, prefix, outdir, epochs, batch_size, lr, test_ratio, categorical_columns, log_columns, integer_columns, tqdm_disable):

    """
    Train a CTABGAN model for driver features
    """

    trainDrivers(csv, prefix, outdir, epochs, batch_size, lr, test_ratio, categorical_columns, log_columns, integer_columns, tqdm_disable)

if __name__ == '__main__':
    trainDriversClick()