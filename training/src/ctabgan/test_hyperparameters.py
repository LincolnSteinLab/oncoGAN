#!/usr/local/bin/python3

# Import modules
import os
import click
from multiprocessing.pool import ThreadPool as Pool
from itertools import product
from ctabgan.train_counts import trainCounts
from ctabgan.train_drivers import trainDrivers

# CLI options
@click.command(name='testHyperparameters')
@click.option('--cpu',
              type=click.INT,
              required=False,
              default=1,
              show_default=True,
              help="Number of CPUs to use")
@click.option("--function",
              type=click.Choice(['counts','drivers'], case_sensitive=False),
              required=True,
              help="Function to test hyperparameters for")
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
              type=click.Tuple([int, int, int]),
              help="A list with an epoch range: start stop step")
@click.option("--batch_size",
              type=click.Tuple([int, int, int]),
              help="A list with a batch_size range: start stop step")
@click.option("--test_ratio",
              type=click.FLOAT,
              default=0.3,
              help="Test ratio")
@click.option("--lr",
              type=click.Tuple([float, float, float]),
              help="A list with a learning rate range: start stop step")
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
def testHyperparameters(cpu, function, csv, prefix, outdir, epochs, batch_size, test_ratio, lr, categorical_columns, log_columns, integer_columns, mixed_columns, general_columns, tqdm_disable):

    """
    Test hyperparameters for counts/drivers CTABGAN models
    """

    # Create the list of options
    options:list = []
    for iproduct in list(product(range(*epochs), range(*batch_size), [i/10000 for i in range(*[int(i*10000) for i in [*lr]])])):
        options.append((csv, prefix, outdir, *iproduct, test_ratio, categorical_columns, log_columns, integer_columns, mixed_columns, general_columns, tqdm_disable))
    
    # Iterate hyperparameters
    click.echo(f"\n########## Testing {len(options)} hyperparameters combinations ##########\n")
    with Pool(cpu) as pool:
        if function == 'counts':
            pool.starmap(trainCounts, options)
        else:
            pool.starmap(trainDrivers, options)
    click.echo(f"\n########################### Done ############################\n")

if __name__ == '__main__':
    testHyperparameters()