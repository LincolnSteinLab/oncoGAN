#!/usr/local/bin/python3

import click
import os
import subprocess
from ctabgan.train_counts import trainCountsClick
from ctabgan.train_mutations import trainMutations
from ctabgan.train_drivers import trainDriversClick
from ctabgan.test_hyperparameters import testHyperparameters

@click.group()
def cli():
    pass

@click.command(name="jupyter")
@click.option('--port',
              type=click.INT,
              default=8890,
              show_default=True,
              help='Port to launch jupyter lab on')
def jupyter(port):
    
    """
    Launches a jupyter lab instance
    """

    subprocess.run(["/bin/bash", "-c", f"jupyter-lab --no-browser --port {port} --ip=$(hostname)"])

cli.add_command(trainCountsClick)
cli.add_command(trainMutations)
cli.add_command(trainDriversClick)
cli.add_command(testHyperparameters)
cli.add_command(jupyter)
if __name__ == '__main__':
    cli()
