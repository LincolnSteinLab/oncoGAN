# The reference genome for chromosomes 20, 21 and 22 is too large to upload to GitHub
# To generate this reference follow these steps:

mkdir reference
tabix PATH_TO_REFERENCE/hg19.fa 20 > reference/hg19_chr20_21_22.fa
tabix PATH_TO_REFERENCE/hg19.fa 21 >> reference/hg19_chr20_21_22.fa
tabix PATH_TO_REFERENCE/hg19.fa 22 >> reference/hg19_chr20_21_22.fa
samtools faidx reference/hg19_chr20_21_22.fa