#!/usr/local/bin/python3

import click
from oncogan_to_fasta import oncogan_to_fasta
from InSilicoSeq import InSilicoSeq
from BAMsurgeon import BAMsurgeon

@click.group()
def cli():
    pass

cli.add_command(oncogan_to_fasta)
cli.add_command(InSilicoSeq)
cli.add_command(BAMsurgeon)
if __name__ == '__main__':
    cli()
