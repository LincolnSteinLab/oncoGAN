#!/usr/local/bin/python3

import click
from InSilicoSeq import InSilicoSeq
from InSilicoSeq_CNA import InSilicoSeq_CNA

@click.group()
def cli():
    pass

cli.add_command(InSilicoSeq)
cli.add_command(InSilicoSeq_CNA)
if __name__ == '__main__':
    cli()
