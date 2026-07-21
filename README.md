[![license](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.com/LincolnSteinLab/oncoGAN/tree/main/LICENSE) ![version](https://badgen.net/badge/version/v1.0.0/blue) ![fasta2bam](https://badgen.net/badge/fasta2bam/v0.1/blue)
 [![zenodo](https://img.shields.io/badge/docs-zenodo-green)](https://zenodo.org/records/13946726) [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13946726.svg)](https://doi.org/10.5281/zenodo.13946726) [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/anderdnavarro/OncoGAN) [![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm-dark.svg)](https://huggingface.co/collections/anderdnavarro/oncogan-67110940dcbafe5f1aa2d524)

# OncoGAN

A pipeline that accurately simulates high quality publicly cancer genomes (VCFs, CNAs and SVs). OncoGAN offers a solution to current challenges in data accessibility and privacy while also serving as a powerful tool for enhancing algorithm development and benchmarking.

## Big update - v1.0.0 - Main changes

We have upgraded the internal generative models from Generative Adversarial Network (GAN)–based architectures ([CTAB-GAN+](https://github.com/Team-TUD/CTAB-GAN-Plus) and [CTGAN](https://docs.sdv.dev/sdv)) to a Flow-Matching Diffusion–based approach ([Calo-Forest](https://github.com/layer6ai-labs/calo-forest)).

This new architecture provides several key improvements:
- Higher accuracy in simulated mutational profiles
- Substantially expanded Single Base Substitutions (SBSs) signature coverage (v0.2.1: 6–8 → v1.0.0: 60)
- Inclusion of indel (IDs) signatures (v0.2.1: 0 → v1.0.0: 17)
- Improved genomic distribution of mutations at resolutions up to 1 Mb
- Enhanced driver mutation profiles

Additionally:
 1. `simulation` and `training` Docker images are now merged into a single, lightweight, and user-friendly image
 2. Copy number alterations and structural variants are now simulated by [SimChA](https://github.com/schwarzlab-ccb/SimChA) and integrated into the OncoGAN workflow automatically

---

## Index

1. [Installation](#installation)
    - [Docker](#docker)
    - [Singularity](#singularity)
    - [Download models](#download-models)
2. [Generate synthetic VCFs](#generate-synthetic-vcfs)
    - [Real profiles](#tumors-with-real-profiles)
    - [Custom profiles](#tumors-with-custom-profiles)
    - [More options](#more-options)
3. [Train new models](#train-new-models)
    - [Calo-Forest command](#calo-forest-command)
    - [DAE command](#dae-command)
4. [DeepTumour](#deeptumour)
5. [Create tumor BAMs](#create-tumor-bams)
    - [Preparation of a tumor FASTA genome](#preparation-of-a-tumor-fasta-genome)
    - [InSilicoSeq](#insilicoseq)
    - [BAMsurgeon](#bamsurgeon)

## Installation

We have created three docker images with all dependencies installed as there are version incompatibility issues between the different modules: 

- OncoGAN -> Environment and scripts used to simulate and train OncoGAN models
- DeepTumour -> Algorithm developed to detect the tumor type of origin based o somatic mutations ([Ref](https://www.nature.com/articles/s41467-019-13825-8))
- fasta2bam -> Module to generate FASTQ/BAM files using OncoGAN's output

However, due to the size of the models, they couldn’t be stored in the Docker images and need to be downloaded separately (*see [Download models](#Download-models) section below*).

### Docker

If you don't have docker already installed in your system, please follow these [instructions](https://docs.docker.com/get-docker/).

```bash
# OncoGAN
docker pull oicr/oncogan:v1.0.0

# DeepTumour
docker pull ghcr.io/lincolnsteinlab/deeptumour:3.0.5

# fasta2bam
docker pull oicr/oncogan:fasta2bam_v0.1
```

### Singularity

If you don't have singularity already installed in your system, please follow these [instructions](https://apptainer.org/admin-docs/master/installation.html).

```bash
# OncoGAN
singularity pull docker://oicr/oncogan:v1.0.0

# DeepTumour
singularity pull docker://ghcr.io/lincolnsteinlab/deeptumour:3.0.5

# fasta2bam
singularity pull docker://oicr/oncogan:fasta2bam_v0.1
```

### Download models

OncoGAN models for the thirty tumor types and DeepTumour models (default and enhanced with OncoGAN v0.2.1 samples) can be found on [HuggingFace](https://huggingface.co/anderdnavarro/OncoGAN) and [Zotero](https://zenodo.org/records/13946726).

## Generate synthetic VCFs

OncoGAN needs two external inputs to simulate new samples:

1. The directory with OncoGAN models downloaded previously
2. **hg19 fasta** reference genome

The output consists of one VCF file (mutations), three TSV files (CNAs, SVs and order of events), and one PNG file (CNAs) per donor, all reported in GRCh38 genomic coordinates by default.

### Tumors with real profiles

```bash
# Docker command
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -v /PATH_TO_HG19_DIR/:/reference \
           -v /PATH_TO_ONCOGAN_MODELS/:/oncoGAN/trained_models \
           -it oicr/oncogan:v1.0.0 \
           vcfGANerator -n 1 --tumor Breast-AdenoCa -r /reference/hs37d5.fa

# Singularity command
singularity exec -H ${pwd}:/home \
            -B /PATH_TO_HG19_DIR/:/reference \
            -B /PATH_TO_ONCOGAN_MODELS/:/oncoGAN/trained_models \
            /PATH_TO/oncogan_v1.0.0.sif launcher.py \
            vcfGANerator -n 1 --tumor Breast-AdenoCa -r /reference/hs37d5.fa
```

The options for the `vcfGANerator` function are:

```bash
vcfGANerator --help

#  Command to simulate mutations (VCF), CNAs and SVs for different tumor types
#  using a Flow-Matching Diffusion model
#
#Options:
#  -@, --cpus INTEGER      Number of CPUs to use  [default: 1]
#  --tumor TEXT            Tumor type to be simulated. Run 'availTumors'
#                          subcommand to check the list of available tumors
#                          that can be simulated
#  -n, --nCases INTEGER    Number of cases to simulate  [default: 1]
#  --NinT FLOAT            Normal in Tumor contamination to be taken into
#                          account when adjusting VAF for CNA-SV events (e.g.
#                          0.20 = 20%)  [default: 0.0]
#  --template PATH         File in CSV format with the number of each type of
#                          mutation to simulate for each donor (template
#                          available on GitHub)
#  -r, --refGenome PATH    hg19 reference genome in fasta format  [required]
#  --prefix TEXT           Prefix to name the output. If not, '--tumor'
#                          option is used as prefix
#  --outDir DIRECTORY      Directory where save the simulations [default: .]
#  --hg19                  Transform the mutations to hg19. [default: hg38]
#  --mut / --no-mut        Simulate mutations  [default: mut]
#  --CNA-SV / --no-CNA-SV  Simulate CNA and SV events  [default: CNA-SV]
#  --plots / --no-plots    Save plots  [default: plots]
#  --version               Show the version and exit
#  --help                  Show this message and exit
```

### Tumors with custom profiles

To generate tumors with custom profiles, users can use the [template](template_custom_simulation.csv), which contains a list of possible mutation types and signatures to simulate. If no CNA-SV are required, use the `--no-CNA-SV` option in the command line.

```bash
# Docker command
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -v /PATH_TO_HG19_DIR/:/reference \
           -v /PATH_TO_ONCOGAN_MODELS/:/oncoGAN/trained_models \
           -it oicr/oncogan:v1.0.0 \
           vcfGANerator --template /home/template_custom_simulation.csv -r /reference/hs37d5.fa

# Singularity command
singularity exec -H ${pwd}:/home \
            -B /PATH_TO_HG19_DIR/:/reference \
            -B /PATH_TO_ONCOGAN_MODELS/:/oncoGAN/trained_models \
            /PATH_TO/oncogan_v1.0.0.sif launcher.py \
            vcfGANerator --template /home/template_custom_simulation.csv -r /reference/hs37d5.fa
```

Among all the options offered by docker (`docker run --help`), we recommend:

- `--rm`: Automatically remove the container when it exits.
- `-u, --user`: Specify the user ID and its group ID. It's useful to not run the pipeline as root.
- `-v, --volume`: Mount local volumes in the container.
  - With the option `-v $(pwd):/home/`, OncoGAN results will be in your current directory.
- `-i, --interactive`: Keep STDIN open even if not attached.
- `-t, --tty`: Allocate a pseudo-TTY. When combined with `-i` it allows you to connect your terminal with the container terminal.

For singularity, the `-H` (for `/home`) and `-B` (for other directories) options are analogous to `-v` docker option.

### More options 

List of available tumors:

```bash
docker run --rm -it oicr/oncogan:v1.0.0 availTumors

# or 

singularity exec /PATH_TO/oncogan_v1.0.0.sif launcher.py availTumors
 
# This is the list of available tumor types that can be simulated using oncoGAN - ('*' Available CNA model):

# Biliary-AdenoCA   Bladder-TCC*      Bone-Leiomyo      Bone-Osteosarc    Breast-AdenoCa*   Cervix-SCC*       
# CNS-GBM*          CNS-Medullo       CNS-Oligo*        CNS-PiloAstro     ColoRect-AdenoCA* Eso-AdenoCa*      
# Head-SCC*         Kidney-ChRCC      Kidney-RCC*       Liver-HCC*        Lung-AdenoCA*     Lung-SCC*         
# Lymph-BNHL        Lymph-CLL         Myeloid-MPN       Ovary-AdenoCA*    Panc-AdenoCA*     Panc-Endocrine    
# Prost-AdenoCA*    Skin-Melanoma*    Stomach-AdenoCA*  Thy-AdenoCA       Uterus-AdenoCA* 
```

>NOTE: A general model is used for tumor types without a specific SimChA model.

## Train new models

Files used to train OncoGAN models can be found on [HuggingFace](https://huggingface.co/datasets/anderdnavarro/OncoGAN-training_files) and [Zotero](https://zenodo.org/records/13946726). The directory containing these files or your custom training files need to be mounted into the docker/singularity container.

We used two different training approaches: 

- [Calo-Forest](https://github.com/layer6ai-labs/calo-forest) -> To train *donor characteristics*, *mutational signatures*, *genomic positions* and *drivers*
- Denoising AutoEncoder (DAE) -> To reduce *genomic positions* training file dimensionality prior to Calo-Forest training (Encoder). During simulation we used the Decoder model to transform back the *genomic positions* from latent space to real genomic length.

### Calo-Forest command

The command to train a Calo-Forest model is:

```bash
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -it oicr/oncogan:v1.0.0 \
           trainCaloForest --csv /home/caloforest_training_test.csv --config /home/caloforest_config_test.json --prefix caloforest_test

# Singularity command
singularity exec -H ${pwd}:/home \
            /PATH_TO/oncogan_v1.0.0.sif launcher.py \
            trainCaloForest --csv /home/caloforest_training_test.csv --config /home/caloforest_config_test.json --prefix caloforest_test
```

### DAE command

We recommend training the DAE model interactively inside the Docker container, using as template the [`dae_training.py` script](src/dae_training.py):

> [!WARNING] 
> Modify line 27 of the `dae_training.py` script with your your training data.

```bash
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           --entrypoint /bin/bash \
           -it oicr/oncogan:v1.0.0

# Activate the specific conda environment
> micromamba activate dae

# Singularity command
singularity shell -H ${pwd}:/home \
            /PATH_TO/oncogan_v1.0.0.sif

# Activate the specific conda environment
> micromamba activate dae

# Then, run the dae_training.py script with your custom configuration interactively
```

## DeepTumour

[DeepTumour](https://www.nature.com/articles/s41467-019-13825-8) is a tool that can predict the tumor type of origin based on the pattern of somatic mutations. We used a second version of this tool, that can predict 29 tumor types instead of 24, to validate that our simulations were correctly assigned to their training tumor type. We also trained a new model using a mix of real and synthetic donors, improving the overall accuracy of the model. Both the original and the new model are available on [HuggingFace](https://huggingface.co/anderdnavarro/DeepTumour) and [Zotero](https://zenodo.org/records/13946726). To use them:

```bash
docker run --rm \
           -u $(id -u):$(id -g) \
           -v $(pwd):/WORKDIR \
           -v /PATH_TO_DEEPTUMOUR_MODEL/:/home/deeptumour/src/trained_models \
           -v /PATH_TO_HG19_DIR/:/reference \
           -it -a stdout -a stderr \
           ghcr.io/lincolnsteinlab/deeptumour:3.0.5 --help

# or

singularity exec \
            -B $(pwd):/WORKDIR \
            -B /PATH_TO_DEEPTUMOUR_MODEL//home/deeptumour/src/trained_models \
            -B /PATH_TO_HG19_DIR/:/reference \
            /PATH_TO/deeptumour_3.0.5.sif --help

# (without the PATH_TO_DEEPTUMOUR_MODEL line, will run the standard DeepTumour model)

# Predict cancer type from a VCF file using DeepTumour

# Options:
#   --vcfFile PATH      VCF file to analyze [Use --vcfFile or --vcfDir]
#   --vcfDir DIRECTORY  Directory with VCF files to analyze [Use --vcfFile or --vcfDir]
#   --reference PATH    hg19 reference genome in fasta format  [required]
#   --hg38              Use this tag if your VCF is in hg38 coordinates
#   --keep_input        Use this tag to also save DeepTumour input as a csv file
#   --outDir DIRECTORY  Directory where save DeepTumour results. Default is the current directory
#   --stdout            Use this tag to print the results to stdout instead of saving them to a file
#   --help              Show this message and exit.
```

## Create tumor BAMs

The `fasta2bam` module is a set of scripts used to generate FASTQ and BAM files from OncoGAN’s output. We provide two different approaches to generate the BAM files:

1. [**InSilicoSeq**](https://insilicoseq.readthedocs.io/en/latest/): A tool that simulates sequencing reads (FASTQs) from a reference/custom genome. Using the [`OncoGAN-to-FASTA` function](#preparation-of-a-tumor-fasta-genome), we incorporate the mutations and CNAs from OncoGAN’s output into a reference FASTA genome to create the tumor genome that will serve as input for InSilicoSeq.
2. [**BAMsurgeon**](https://github.com/adamewing/bamsurgeon/tree/master): A tool that modifies existing BAMs to add synthetic mutations, CNAs and SVs.

Example files for testing are available in the [`bam_implementation` folder](bam_implementation/test).

> **Information**: If you need assistance integrating OncoGAN’s output with other tools, feel free to open an issue.

### Preparation of a tumor FASTA genome

To use InSilicoSeq, we first need to create a tumor FASTA genome that incorporates the mutations and CNAs from OncoGAN’s output, preserving the simulated order of events. The output is a FASTA genome that can also be used with any other FASTA-to-FASTAQ generator tool. This is accomplished using the `OncoGAN-to-FASTA` function:

```bash
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -v /PATH_TO_HG19_OR_HG38_DIR/:/reference \
           -v /PATH_TO_FILES/:/files \
           -it oicr/oncogan:fasta2bam_v0.1 \
           OncoGAN-to-FASTA --input files/test_mutations_cna_adjusted.vcf \
                            --reference_genome files/reference/hg19_chr1_1_20000000.fa \
                            --events files/test_events_order.tsv \
                            --sv files/test_sv.tsv \
                            --dbSNP files/test_dbSNP_germline.vcf \
                            --out_dir insilicoseq_test_cna

# or

singularity exec -H ${pwd}:/home \
            -B /PATH_TO_HG19_OR_HG38_DIR/:/reference \
            -B /PATH_TO_FILES/:/files \
            /PATH_TO/oncogan_fasta2bam_v0.1.sif /usr/local/bin/_entrypoint.sh python3 /src/launcher.py \
            OncoGAN-to-FASTA --input test/test_mutations_cna_adjusted.vcf \
                             --reference_genome test/reference/hg19_chr1_1_20000000.fa \
                             --events test/test_events_order.tsv \
                             --sv test/test_sv.tsv \
                             --dbSNP test/test_dbSNP_germline.vcf \
                             --out_dir insilicoseq_test_cna
```

### InSilicoSeq

Once the custom FASTA genome is created, we can use InSilicoSeq wrapper to generate the FASTQ files. Then, they can be aligned to the reference genome using any aligner (e.g. BWA, Bowtie2, etc.) to create the BAM files.

```bash
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -v /PATH_TO_HG19_OR_HG38_DIR/:/reference \
           -v /PATH_TO_CUSTOM_FASTA/:/fasta \
           -it oicr/oncogan:fasta2bam_v0.1 \
           InSilicoSeq --cpus 20 \
                       --prefix insilicoseq_test_sim1 \
                       --custom_genome fasta/sim1_genome.fa \
                       --reference_genome reference/hg19_chr1_1_20000000.fa \
                       --out_dir insilicoseq_test_cna \
                       --coverage 30

# or

singularity exec -H ${pwd}:/home \
            -B /PATH_TO_HG19_OR_HG38_DIR/:/reference \
            -B /PATH_TO_CUSTOM_FASTA/:/fasta \
            /PATH_TO/oncogan_fasta2bam_v0.1.sif /usr/local/bin/_entrypoint.sh python3 /src/launcher.py \
            InSilicoSeq --cpus 20 \
                       --prefix insilicoseq_test_sim1 \
                       --custom_genome fasta/sim1_genome.fa \
                       --reference_genome reference/hg19_chr1_1_20000000.fa \
                       --out_dir insilicoseq_test_cna \
                       --coverage 30
```

### BAMsurgeon

BAMsurgeon requires a reference BAM file in which mutations will be inserted. Ideally, users should provide their own BAM file for this purpose. However, if no real BAM is available, you can generate one by first creating a custom FASTA genome containing only germline variants using the `OncoGAN-to-FASTA` function. This FASTA can then be used as input for `InSilicoSeq` to simulate a normal BAM file.

```bash
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -v /PATH_TO_FILES/:/files \
           -v /REFERENCE_BAMS/:/bams \
           -it oicr/oncogan:fasta2bam_v0.1 \
           BAMsurgeon --cpus 20 \
                      --varfile files/test_mutations_cna_adjusted_bamsurgeon.vcf \
                      --sv_varfile files/test_sv.tsv \
                      --cnv_varfile files/test_cna.tsv \
                      --bamfile bams/test_normal_bam/hg19_normal.bam \
                      --donorbam bams/test_donorbam/hg19_donorbam.bam \
                      --reference files/reference/hg19_chr1_1_20000000.fa \
                      --prefix bamsurgeon_test_sim1 \
                      --out_dir bamsurgeon_test_cna

# or

singularity exec -H ${pwd}:/home \
            -B /PATH_TO_FILES/:/files \
            -B /REFERENCE_BAMS/:/bams \
            /PATH_TO/oncogan_fasta2bam_v0.1.sif /usr/local/bin/_entrypoint.sh python3 /src/launcher.py \
            BAMsurgeon --cpus 20 \
                       --varfile files/test_mutations_cna_adjusted_bamsurgeon.vcf \
                       --sv_varfile files/test_sv.tsv \
                       --cnv_varfile files/test_cna.tsv \
                       --bamfile bams/test_normal_bam/hg19_normal.bam \
                       --donorbam bams/test_donorbam/hg19_donorbam.bam \
                       --reference files/reference/hg19_chr1_1_20000000.fa \
                       --prefix bamsurgeon_test_sim1 \
                       --out_dir bamsurgeon_test_cna
```

>Note: The `--donorbam` option in BAMSurgeon is used to extract additional reads for CNA duplications.