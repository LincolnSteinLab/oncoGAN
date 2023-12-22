#!/usr/local/bin/python3

import os
import click
import pandas as pd

@click.command(name="mergeVCFs")
@click.option("--vcfDir", "vcfDir",
              type=click.Path(exists=True, file_okay=False),
              required=True,
              help="Directory where VCFs are stored")
@click.option("--prefix",
              type=click.STRING,
              default="merged",
              help="Prefix to name the output")
@click.option("--outDir", "outDir",
              type=click.Path(exists=True, file_okay=False),
              default=os.getcwd(),
              show_default=False,
              help="Directory where save the merge CSV file. Default is the current directory")
def mergeVCFs(vcfDir, prefix, outDir):

    """
    Merge all VCF files in a directory into a single CSV file
    """

    vcf_files = [os.path.join(vcfDir, f) for f in os.listdir(vcfDir) if f.endswith(".vcf")]
    
    # Open and concatenate all VCF files
    merged_df = pd.DataFrame()
    for file in vcf_files:
        vcf_df = pd.read_csv(file, delimiter='\t', comment='#', header=None)
        merged_df = pd.concat([merged_df, vcf_df])

    # Add column names to the merged DataFrame
    merged_df.columns = ['CHROM','POS','ID','REF','ALT','VAF','MUT']

    # Sort by ID column
    merged_df.sort_values(by=['ID','CHROM','POS'], inplace=True)

    # Save the merged DataFrame to a CSV file
    output_file = os.path.join(outDir, f"{prefix}.csv")
    merged_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    mergeVCFs()