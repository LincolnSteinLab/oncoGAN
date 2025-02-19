#!/usr/local/bin/python3

import click
from oncoGAN import availTumors, oncoGAN, oncoGAN_custom
from mergeVCFs import mergeVCFs

@click.group()
def cli():
    pass

cli.add_command(availTumors)
cli.add_command(oncoGAN)
cli.add_command(oncoGAN_custom)
cli.add_command(mergeVCFs)
if __name__ == '__main__':
    cli()
