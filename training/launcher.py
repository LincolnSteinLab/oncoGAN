#!/usr/local/bin/python3

import click
import os
import subprocess
from ctabgan.train_counts import training as trainCounts
from ctabgan.train_mutations import training as trainMutations

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

cli.add_command(trainCounts)
cli.add_command(trainMutations)
cli.add_command(jupyter)
if __name__ == '__main__':
    cli()
