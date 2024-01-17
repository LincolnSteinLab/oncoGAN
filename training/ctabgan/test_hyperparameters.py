#!/usr/local/bin/python3

# Import modules
import os
import click
import subprocess
import multiprocessing
from itertools import product

def runScript(script, csv, prefix, outdir, epochs, batch_size, lr, debug):
    if debug:
        subprocess.run(["/bin/bash", "-c", f"{script} --csv {csv} --prefix {prefix} --outdir {outdir} --epochs {epochs} --batch_size {batch_size} --lr {lr}"])
    else:
        subprocess.run(["/bin/bash", "-c", f"{script} --csv {csv} --prefix {prefix} --outdir {outdir} --epochs {epochs} --batch_size {batch_size} --lr {lr} --no-tqdm"])

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
@click.option("--lr",
              type=click.Tuple([float, float, float]),
              help="A list with a learning rate range: start stop step")
@click.option("--debug",
              is_flag=True,
              help="Greater verbosity for debugging purposes")
def testHyperparameters(cpu, function, csv, prefix, outdir, epochs, batch_size, lr, debug):

    """
    Test hyperparameters for the CTABGAN models
    """

    # This function only works for Counts and Drivers
    if function == 'counts':
        script:os.path = '/genomeGAN/ctabgan/train_counts.py'
    elif function == 'drivers':
        script:os.path = '/genomeGAN/ctabgan/train_drivers.py'

    # Create the list of options
    options:list = []
    for iproduct in list(product(range(*epochs), range(*batch_size), [i/10000 for i in range(*[int(i*10000) for i in [*lr]])])):
        options.append(tuple([script, csv, prefix, outdir]+list(iproduct)+[debug]))
    
    # Iterate hyperparameters
    click.echo(f"\n########## Testing {len(options)} hyperparameters combinations ##########\n")
    with multiprocessing.Pool(cpu) as pool:
        pool.starmap(runScript, options)
    click.echo(f"                 ########## Done ##########                     \n")

if __name__ == '__main__':
    testHyperparameters()