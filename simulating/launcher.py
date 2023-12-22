#!/usr/local/bin/python3

import click
from genomeGAN import availTumors, genomeGAN
from mergeVCFs import mergeVCFs

@click.group()
def cli():
    pass

cli.add_command(availTumors)
cli.add_command(genomeGAN)
cli.add_command(mergeVCFs)
if __name__ == '__main__':
    cli()
