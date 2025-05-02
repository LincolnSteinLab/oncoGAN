import os
import shutil
import click
import subprocess
import pandas as pd
from oncogan_to_fasta import read_vcf

def convert_vcf_to_bed(vcf_file:click.Path) -> tuple:

    """
    Convert VCF to BED format
    """

    # Open the VCF
    vcf:pd.DataFrame = read_vcf(vcf_file)
    vcf['end'] = vcf.apply(lambda row: int(row['pos']) + (len(row['ref']) + len(row['alt']) - 2), axis=1)
    vcf = vcf[['chrom', 'pos', 'end', 'af', 'alt']]

    # SNV
    vcf_snv:pd.DataFrame = vcf[vcf['pos'] == vcf['end']]
    bed_snv_path:click.Path = f"{os.path.splitext(vcf_file)[0]}_snv.bed"
    vcf_snv.to_csv(bed_snv_path, sep="\t", index=False, header=False)
    
    # Indel
    vcf_indel:pd.DataFrame = vcf[vcf['pos'] != vcf['end']]
    vcf_indel['type'] = vcf_indel.apply(lambda row: 'INS' if len(row['alt']) > 1 else 'DEL', axis=1)
    vcf_indel['alt'] = vcf_indel.apply(lambda row: row['alt'][1:] if row['type'] == 'INS' else '.', axis=1)
    vcf_indel = vcf_indel[['chrom', 'pos', 'end', 'af', 'type', 'alt']]
    bed_indel_path:click.Path = f"{os.path.splitext(vcf_file)[0]}_indel.bed"
    vcf_indel.to_csv(bed_indel_path, sep="\t", index=False, header=False)

    return(bed_snv_path, bed_indel_path)

def convert_cna_to_bed(cna_file:click.Path) -> click.Path:

    """
    Convert CNA file to BED format
    """

    # Open the CNA file
    cna:pd.DataFrame = pd.read_csv(cna_file, sep="\t")
    cna['cn'] = cna.apply(lambda row: int(row['major_cn']) + int(row['minor_cn']), axis=1)
    cna = cna[['chrom', 'start', 'end', 'cn']]
    
    # Convert to BED format
    bed_cna_path:click.Path = f"{os.path.splitext(cna_file)[0]}.bed"
    cna.to_csv(bed_cna_path, sep="\t", index=False, header=False)

    # Tabix index the BED file
    cmd:list = ["bgzip", bed_cna_path]
    subprocess.run(cmd, check=True)
    cmd:list = ["tabix", "-p", "bed", f"{bed_cna_path}.gz"]
    subprocess.run(cmd, check=True)
    bed_cna_path = f"{bed_cna_path}.gz"

    return(bed_cna_path)

@click.command(name="BAMsurgeon")
@click.option("-@", "--cpus",
              type=click.INT,
              default=1,
              show_default=True,
              help="Number of CPUs to use")
@click.option("-v", "--varfile",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="OncoGAN VCF mutations")
@click.option("-f", "--bamfile",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="SAM/BAM file from which to obtain reads")
@click.option("-r", "--reference",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="Reference genome, fasta indexed with bwa index and samtools faidx, if not it will generate the index")
@click.option("-p", "--prefix",
              type=click.STRING,
              required=True,
              help="BAM file prefix")
@click.option("-o", "--out_dir",
              type=click.Path(exists=False, file_okay=False),
              default=os.getcwd(),
              show_default=False,
              help="Directory where save the BAMs. Default is the current directory")
@click.option("-s", "--snvfrac",
              type=click.FLOAT,
              default=1.0,
              show_default=True,
              help="Maximum allowable linked SNP MAF (for avoiding haplotypes)")
@click.option("-m", "--mutfrac",
              type=click.FLOAT,
              default=0.5,
              show_default=True,
              help="Allelic fraction at which to make SNVs")
@click.option("-c", "--cnvfile",
              type=click.Path(exists=True, file_okay=True),
              required=False,
              help="TSV containing CNAs simulated with OncoGAN")
@click.option("-d", "--coverdiff",
              type=click.FLOAT,
              default=0.9,
              show_default=True,
              help="Allow difference in input and output coverage")
def BAMsurgeon(cpus, varfile, bamfile, reference, prefix, out_dir, snvfrac, mutfrac, cnvfile, coverdiff):
    
    """Run BAMsurgeon"""

    # Check if the genome is indexed
    if not os.path.exists(reference + ".fai"):
        print(f"Indexing {os.path.basename(reference)} with samtools faidx")
        cmd:list = ["samtools", "faidx", reference]
        subprocess.run(cmd, check=True)
    if not os.path.exists(reference + ".bwt"):
        print(f"Indexing {os.path.basename(reference)} with bwa index")
        cmd:list = ["bwa", "index", reference]
        subprocess.run(cmd, check=True)

    # Check if the BAM file is indexed
    if not os.path.exists(bamfile + ".bai"):
        print(f"Indexing {os.path.basename(bamfile)} with samtools index")
        cmd:list = ["samtools", "index", bamfile]
        subprocess.run(cmd, check=True)

    # Convert VCF to BED required format
    bed_snv_path, bed_indel_path = convert_vcf_to_bed(varfile)

    # Convert CNA file to BED required format
    if cnvfile is not None:
        bed_cna_path:click.Path = convert_cna_to_bed(cnvfile)

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    outbam_snv:click.Path = f"{os.path.join(out_dir, prefix)}_snv.bam"
    outbam_indel:click.Path = f"{os.path.join(out_dir, prefix)}_snv_indel.bam"
    outbam_sorted_indel:click.Path = f"{os.path.join(out_dir, prefix)}_snv_indel.sorted.bam"

    # SNV command
    cmd:list = [
        "addsnv.py",
        "--procs", f"{cpus}",
        "--varfile", bed_snv_path,
        "--bamfile", bamfile,
        "--reference", reference,
        "--outbam", outbam_snv,
        "--snvfrac", str(snvfrac),
        "--mutfrac", str(mutfrac),
        "--coverdiff", str(coverdiff)
    ]
    
    if cnvfile is not None:
        cmd.extend(["--cnvfile", bed_cna_path])
    
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

    ## Clean
    os.rename(f'{prefix}_snv.addsnv.{os.path.splitext(os.path.basename(varfile))[0]}_snv.vcf', os.path.join(out_dir, f"{prefix}_addsnv.vcf"))
    shutil.move(f'addsnv_logs_{prefix}_snv.bam', os.path.join(out_dir, f"{prefix}_addsnv_logs"))
    os.rmdir('addsnv.tmp')

    # Index SNV BAM
    cmd:list = ["samtools", "index", outbam_snv]
    subprocess.run(cmd, check=True)

    # Indel command
    cmd:list = [
        "addindel.py",
        "--procs", f"{cpus}",
        "--varfile", bed_indel_path,
        "--bamfile", outbam_snv,
        "--reference", reference,
        "--outbam", outbam_indel,
        "--snvfrac", str(snvfrac),
        "--mutfrac", str(mutfrac),
        "--coverdiff", str(coverdiff)
    ]
    
    if cnvfile is not None:
        cmd.extend(["--cnvfile", bed_cna_path])
    
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

    # Index SNV+Indel BAM
    cmd:list = ["samtools", "sort", "-O", "BAM", "-o", outbam_sorted_indel, outbam_indel]
    subprocess.run(cmd, check=True)
    cmd:list = ["samtools", "index", outbam_sorted_indel]
    subprocess.run(cmd, check=True)

    ## Clean
    os.rename(f'{prefix}_snv_indel.addindel.{os.path.splitext(os.path.basename(varfile))[0]}_indel.vcf', os.path.join(out_dir, f"{prefix}_addindel.vcf"))
    shutil.move(f'addindel_logs_{prefix}_snv_indel.bam', os.path.join(out_dir, f"{prefix}_addindel_logs"))
    os.rmdir('addindel.tmp')
    os.remove(outbam_snv)
    os.remove(outbam_snv + ".bai")
    os.remove(outbam_indel)
