import os
import shutil
import click
import subprocess
import pandas as pd
from pyfaidx import Fasta
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

def convert_sv_to_bed(sv_file:click.Path, cna_file:click.Path, reference:click.Path, bamfile:click.Path) -> tuple:

    """
    Convert SV file to BED format
    """

    # Open reference genome
    genome:Fasta = Fasta(reference)

    # Open the SV and CNA files
    sv:pd.DataFrame = pd.read_csv(sv_file, sep="\t")
    cna:pd.DataFrame = pd.read_csv(cna_file, sep="\t")

    # Calculate total CN
    cna['total_cn'] = cna.apply(lambda row: int(row['major_cn']) + int(row['minor_cn']) - (1 if row['minor_cn'] == 0 else 2), axis=1)

    # DUP and DEL
    sv_dup_del:pd.DataFrame = sv[sv['svclass'].isin(['DUP', 'DEL'])].reset_index(drop=True)
    if not sv_dup_del.empty:
        sv_dup_del = sv_dup_del[['chrom1', 'start1', 'start2', 'svclass', 'cna_id']]
        sv_dup_del = sv_dup_del.drop_duplicates()
        sv_dup_del = sv_dup_del.merge(cna[['cna_id', 'total_cn']], on='cna_id', how='left')
        sv_dup_del['total_cn'] = sv_dup_del.apply(lambda row: '' if row['svclass'] == 'DEL' else row['total_cn'], axis=1)
        sv_dup_del = sv_dup_del.drop(columns=['cna_id'])

        ## Adjust CNA boundaries to avoid 'N' sequences (telomeres and centromeres) and border segments with no reads
        for i,row in sv_dup_del.iterrows():
            sequence = str(genome[str(row['chrom1'])][int(row['start1']):int(row['start2'])])
            leading_Ns = len(sequence) - len(sequence.lstrip('N'))
            trailing_Ns = len(sequence) - len(sequence.rstrip('N'))
            new_start1 = row['start1']+leading_Ns
            new_start2 = row['start2']-trailing_Ns
            sv_dup_del.loc[i,'start2'] = new_start2

            cmd:list = [
                "samtools", "depth",
                "-r", f"{row['chrom1']}:{new_start1}-{new_start1+10000}",
                "-a",
                "--reference", reference,
                bamfile
            ]
            depth1 = subprocess.run(cmd, check=True, capture_output=True, text=True)
            for line in depth1.stdout.strip().split('\n'):
                chrom, pos, cov = line.split('\t')
                if int(cov) < 15:
                    continue
                else:
                    sv_dup_del.loc[i,'start1'] = pos
                    break

            cmd:list = [
                "samtools", "depth",
                "-r", f"{row['chrom1']}:{new_start2-10000}-{new_start2}",
                "-a",
                "--reference", reference,
                bamfile
            ]
            depth2 = subprocess.run(cmd, check=True, capture_output=True, text=True)
            for line in reversed(depth2.stdout.strip().split('\n')):
                chrom, pos, cov = line.split('\t')
                if int(cov) < 15:
                    continue
                else:
                    sv_dup_del.loc[i,'start2'] = pos
                    break

        ## Save the BED file
        bed_sv_dup_del_path:click.Path = f"{os.path.splitext(sv_file)[0]}_dup_del.bed"
        sv_dup_del.to_csv(bed_sv_dup_del_path, sep="\t", index=False, header=False)
    else:
        bed_sv_dup_del_path = None

    # INV
    sv_inv:pd.DataFrame = sv[sv['svclass'].isin(['h2hINV', 't2tINV'])].reset_index(drop=True)
    if not sv_inv.empty:
        sv_inv = sv_inv[['chrom1', 'start1', 'start2', 'svclass']]
        sv_inv['svclass'] = 'INV'
        sv_inv = sv_inv.rename(columns={'chrom1': 'x1', 'start1': 'x2', 'start2': 'x3', 'svclass': 'x4'})
    else:
        sv_inv = None

    # TRA
    sv_tra:pd.DataFrame = sv[sv['svclass'] == 'TRA'].reset_index(drop=True)
    if not sv_tra.empty:
        sv_tra['svclass'] = 'TRN'
        sv_tra['strand'] = sv_tra.apply(lambda row: f"{row['strand1']}{row['strand2']}", axis=1)
        sv_tra = sv_tra[['chrom1', 'start1', 'end1', 'svclass', 'chrom2', 'start2', 'end2', 'strand']]
        sv_tra = sv_tra.rename(columns={
            'chrom1': 'x1', 'start1': 'x2', 'end1': 'x3', 'svclass': 'x4',
            'chrom2': 'x5', 'start2': 'x6', 'end2': 'x7', ' strand': 'x8'})
    else:
        sv_tra = None

    ## Combine INV and TRA
    if sv_inv is not None and sv_tra is not None:
        sv_inv_tra:pd.DataFrame = pd.concat([sv_inv, sv_tra], ignore_index=True)
        sv_inv_tra = sv_inv_tra.fillna('')
    elif sv_inv is not None and sv_tra is None:
        sv_inv_tra = sv_inv
    elif sv_tra is not None and sv_inv is None:
        sv_inv_tra = sv_tra
    else:
        bed_sv_inv_tra_path = None

    ## Save the BED file
    if bed_sv_inv_tra_path is not None:
        bed_sv_inv_tra_path:click.Path = f"{os.path.splitext(sv_file)[0]}_inv_tra.bed"
        sv_inv_tra.to_csv(bed_sv_inv_tra_path, sep="\t", index=False, header=False)

    return(bed_sv_dup_del_path, bed_sv_inv_tra_path)

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
@click.option("-sv", "--sv_varfile",
              type=click.Path(exists=True, file_okay=True),
              required=False,
              help="OncoGAN SV file")
@click.option("-c", "--cnv_varfile",
              type=click.Path(exists=True, file_okay=True),
              required=False,
              help="TSV containing CNAs simulated with OncoGAN")
@click.option("-f", "--bamfile",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="SAM/BAM file from which to obtain reads")
@click.option("--donorbam",
              type=click.Path(exists=True, file_okay=True),
              required=False,
              help="BAM file for donor reads if using BIGDUP (>10kb) mutations")
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
@click.option("--svfrac",
              type=click.FLOAT,
              default=1.0,
              show_default=True,
              help="Allele fraction of variant")
@click.option("-d", "--coverdiff",
              type=click.FLOAT,
              default=0.9,
              show_default=True,
              help="Allow difference in input and output coverage")
def BAMsurgeon(cpus, varfile, sv_varfile, cnv_varfile,  bamfile, donorbam, reference, prefix, out_dir, snvfrac, mutfrac, svfrac, coverdiff):
    
    """Run BAMsurgeon"""

    # If sv_varfile is provided, cnv_varfile must also be provided
    if sv_varfile is not None and cnv_varfile is None:
        raise click.BadParameter("If --sv_varfile is provided, --cnv_varfile must also be provided")
    if sv_varfile is None and cnv_varfile is not None:
        raise click.BadParameter("If --cnv_varfile is provided, --sv_varfile must also be provided")

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

    # Convert SV file to the required format
    if sv_varfile is not None:
        bed_sv_dup_del_path, bed_sv_inv_tra_path = convert_sv_to_bed(sv_varfile, cnv_varfile, reference, bamfile)

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    bamfile_cna:click.Path = f"{os.path.join(out_dir, prefix)}_cna.bam"
    bamfile_sorted_cna:click.Path = f"{os.path.join(out_dir, prefix)}_cna.sorted.bam"
    outbam_snv:click.Path = f"{os.path.join(out_dir, prefix)}_snv.bam"
    outbam_indel:click.Path = f"{os.path.join(out_dir, prefix)}_snv_indel.bam"
    outbam_sorted_indel:click.Path = f"{os.path.join(out_dir, prefix)}_snv_indel.sorted.bam"
    outbam_sv:click.Path = f"{os.path.join(out_dir, prefix)}_snv_indel_sv.bam"
    outbam_sorted_sv:click.Path = f"{os.path.join(out_dir, prefix)}_snv_indel_sv.sorted.bam"

    # CNA command
    if bed_sv_dup_del_path is not None:
        cmd:list = [
            "addsv.py",
            "--procs", f"{cpus}",
            "--varfile", bed_sv_dup_del_path,
            "--bamfile", bamfile,
            "--reference", reference,
            "--outbam", bamfile_cna,
            "--svfrac", str(svfrac)
        ]

        if donorbam is not None:
            cmd.extend(["--donorbam", donorbam])
    
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

        # Index CNA BAM
        cmd:list = ["samtools", "sort", "-O", "BAM", "-o", bamfile_sorted_cna, bamfile_cna]
        subprocess.run(cmd, check=True)
        cmd:list = ["samtools", "index", bamfile_sorted_cna]
        subprocess.run(cmd, check=True)

        ## Clean
        os.rename(f'{prefix}_cna.addsv.{os.path.splitext(os.path.basename(sv_varfile))[0]}_dup_del.vcf', os.path.join(out_dir, f"{prefix}_addsv_dup_del.vcf"))
        shutil.move(f'addsv_logs_{prefix}_cna.bam', os.path.join(out_dir, f"{prefix}_addsv_logs"))
        os.rmdir('addsv.tmp')
        os.remove(bamfile_cna)

    # SNV command
    cmd:list = [
        "addsnv.py",
        "--procs", f"{cpus}",
        "--varfile", bed_snv_path,
        "--bamfile", bamfile if sv_varfile is None else bamfile_sorted_cna,
        "--reference", reference,
        "--outbam", outbam_snv,
        "--snvfrac", str(snvfrac),
        "--mutfrac", str(mutfrac),
        "--coverdiff", str(coverdiff)
    ]
    
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

    # Index SNV BAM
    cmd:list = ["samtools", "index", outbam_snv]
    subprocess.run(cmd, check=True)

    ## Clean
    os.rename(f'{prefix}_snv.addsnv.{os.path.splitext(os.path.basename(varfile))[0]}_snv.vcf', os.path.join(out_dir, f"{prefix}_addsnv.vcf"))
    shutil.move(f'addsnv_logs_{prefix}_snv.bam', os.path.join(out_dir, f"{prefix}_addsnv_logs"))
    os.rmdir('addsnv.tmp')
    if sv_varfile is not None:
        os.remove(bamfile_sorted_cna)
        os.remove(bamfile_sorted_cna + ".bai")

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

    # SV command
    if bed_sv_inv_tra_path is not None:
        cmd:list = [
            "addsv.py",
            "--procs", f"{cpus}",
            "--varfile", bed_sv_inv_tra_path,
            "--bamfile", outbam_sorted_indel,
            "--reference", reference,
            "--outbam", outbam_sv,
            "--svfrac", str(svfrac)
        ]
    
        print(' '.join(cmd))
        subprocess.run(cmd, check=True)

        # Index SNV+Indel BAM
        cmd:list = ["samtools", "sort", "-O", "BAM", "-o", outbam_sorted_sv, outbam_sv]
        subprocess.run(cmd, check=True)
        cmd:list = ["samtools", "index", outbam_sorted_sv]
        subprocess.run(cmd, check=True)

        ## Clean
        os.rename(f'{prefix}_cna.addsv.{os.path.splitext(os.path.basename(sv_varfile))[0]}_inv_tra.vcf', os.path.join(out_dir, f"{prefix}_addsv_inv_tra.vcf"))
        shutil.move(f'addsv_logs_{prefix}_snv_indel_sv.bam', os.path.join(out_dir, f"{prefix}_addsv_logs"))
        os.rmdir('addsv.tmp')
        os.remove(outbam_sorted_indel)
        os.remove(outbam_sorted_indel + ".bai")
        os.remove(outbam_sv)
    
    # Merge BAMsurgeon VCFs
    vcf_files:list = [f for f in os.listdir(out_dir) if f.startswith(prefix) and f.endswith('.vcf')]
    vcf_files:list = [os.path.join(out_dir, f) for f in vcf_files]
    merged_vcf:click.Path = os.path.join(out_dir, f"{prefix}_bamsurgeon_output_merged.vcf")
    merged_sorted_vcf:click.Path = os.path.join(out_dir, f"{prefix}_bamsurgeon_output_merged.sorted.vcf")

    ## Header
    with open(merged_vcf, 'w') as out_f:
        with open(vcf_files[0], 'r') as in_f:
            for line in in_f:
                if line.startswith('#'):
                    out_f.write(line)

    ## Mutations
    with open(merged_vcf, 'a') as out_f:
        for vcf in vcf_files:
            with open(vcf, 'r') as in_f:
                for line in in_f:
                    if not line.startswith('#'):
                        out_f.write(line)

    ## Sort
    cmd:list = ["bcftools", "sort", "-o", merged_sorted_vcf, merged_vcf]
    subprocess.run(cmd, check=True)

    ## Remove VCFs
    for vcf in vcf_files:
        os.remove(vcf)
    os.remove(merged_vcf)