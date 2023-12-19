#!/DeepTumour/venvDeepTumour/bin/python

# Read fasta File
import time

from Bio import SeqIO
import pandas as pd

import numpy as np
import allel
import re
from multiprocessing import Pool
import concurrent.futures
import os
import sys
import gzip


def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    reverse_complement = "".join(complement.get(base, base)
                                 for base in reversed(seq))
    return(reverse_complement)


def readAutoChrFASTA():
    x = gzip.open(r'/DeepTumour/references/chromosome.txt.gz', 'rt', encoding='utf-8')
    seq_list = {}
    for i in range(23):
        line = x.readline()
        if(i < 9):
            seq_list[line[0:4]] = line[5:]
        else:
            seq_list[line[0:5]] = line[6:]
    x.close
    return(seq_list)


def vcf2df(flnm):
    # read VCF files
    # filter variants in chr1-chr22
    # count only SNVs

    df = allel.vcf_to_dataframe(flnm, fields='*', alt_number=2)
    print("--- %s Total Variants in the VCF file ---" % len(df))

    if (df['CHROM'][0].startswith('chr')):
        pass
    else:
        df['CHROM'] = ["chr" + str(ch) for ch in df['CHROM']]

    chr_list = ["chr" + str(ch) for ch in range(1, 23)]
    vcf_df = df[(df['is_snp'] == True) & (df['CHROM'].isin(chr_list))]
    print("--- %s SNVs in the VCF file ---" % len(vcf_df))
    return(vcf_df)


def vcf2bins(tmp1, sample_name):
    # set up data frame for SNV counts
    # load header
    filename = '/DeepTumour/references/hg19.1Mb.header.gz'
    binhd_df = pd.read_csv(filename, compression='gzip', header=None)

    # initial dataframe

    test = tmp1.CHROM + '.' + \
        tmp1.POS.apply(lambda x: int(round(float(x) / 1000000))).astype(str)

    bindf_tmp = pd.DataFrame(
        {"bins": pd.Series(pd.Categorical(test, categories=binhd_df.iloc[:, 0]))})

    bins_df = bindf_tmp.groupby('bins').size().reset_index(name=sample_name)

    return bins_df


def df2mut(tmp1, seq_list, sample_name):
    # Mutation types

    # Setup Dataframe

    # load header
    mut_df_header = pd.read_csv(
        '/DeepTumour/references/Mut-Type-Header.csv')

    # initial dataframe
    changes = []
    ref_vcf = []
    ref_hg19 = []
    tmp1 = tmp1.reset_index()
    tmp1 = tmp1.to_numpy()
    for i in range(len(tmp1)):
        ref = tmp1[i, 4]
        ch = tmp1[i, 1]
        st = tmp1[i, 2]
        alt = tmp1[i, 5]
        ref_vcf.append(ref)
        ref_base = seq_list[ch][st - 1].upper()
        ref_hg19.append(ref_base)
        ref_cont = seq_list[ch][st - 2:st + 1].upper()

        if(not ref == ref_base):
            print("Something wrong with the Genome Version, reference bases from VCF file doesn't match with records on hg19 genome\n")
            print(ch)
            print(st)
            print(ref)
            print(ref_base)
            print(ref_cont)

        if (re.search('[GT]', ref)):
            ref = reverse_complement(ref)
            ref_cont = reverse_complement(ref_cont)
            alt = reverse_complement(alt)

        change_sgl = ref + ".." + alt
        change_di1 = ref_cont[0:2] + ".." + ref_cont[0] + alt
        change_di2 = ref_cont[1:3] + ".." + alt + ref_cont[2]
        change_tri = ref_cont + ".." + ref_cont[0] + alt + ref_cont[2]
        changes.append(change_sgl)
        changes.append(change_di1)
        changes.append(change_di2)
        changes.append(change_tri)

    mutdf_tmp = pd.DataFrame({"bins": pd.Series(
        pd.Categorical(changes, categories=mut_df_header.iloc[:, 0]))})

    mut_df = mutdf_tmp.groupby('bins').size().reset_index(name=sample_name)
    if sum(mut_df[sample_name]) > 0:
        tmp_sgl = mut_df[sample_name].iloc[0:6] / sum(mut_df[sample_name].iloc[0:6])
        tmp_di = mut_df[sample_name].iloc[6:54] / sum(mut_df[sample_name].iloc[6:54])
        tmp_tri = mut_df[sample_name].iloc[54:150] / sum(mut_df[sample_name].iloc[54:150])
        mut_df[sample_name].iloc[0:6] = tmp_sgl
        mut_df[sample_name].iloc[6:54] = tmp_di
        mut_df[sample_name].iloc[54:150] = tmp_tri
    return(mut_df)


if __name__ == '__main__':
    # load FASTA Files:
    start_time1 = time.time()
    seq_list = readAutoChrFASTA()
    print("--- %s seconds to load %s FASTA Files---" %
          ((time.time() - start_time1), len(seq_list)))

    # load VCF file:
    vcf_df = vcf2df(sys.argv[1])
    sample_name = sys.argv[2]

    # convert VCF to Bin counts
    start_time = time.time()
    bins_df = vcf2bins(vcf_df, sample_name)
    print(bins_df.shape)
    print("--- %s seconds to make SNV Counts Data Frame ---" %
          (time.time() - start_time))

    # convert VCF to Mutation Types
    start_time = time.time()
    mut_df = df2mut(vcf_df, seq_list, sample_name)

    print("--- %s seconds to make Mutation Type Data Frame ---" %
          (time.time() - start_time))

    cb_df = pd.concat([bins_df, mut_df])

    cb_df = cb_df.set_index('bins')

    cb_df = cb_df.transpose()
    cb_df.to_csv(sys.argv[1] + '.csv', header=True)

    print("total --- %s seconds ---" % (time.time() - start_time1))
