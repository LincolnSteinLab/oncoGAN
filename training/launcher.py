#!/usr/local/bin/python3

import click
from ctabgan.train_counts import training as trainCounts
from ctabgan.train_mutations import training as trainMutations

@click.group()
def cli():
    pass

cli.add_command(trainCounts)
cli.add_command(trainMutations)
if __name__ == '__main__':
    cli()
