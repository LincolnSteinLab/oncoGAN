[![license](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.com/LincolnSteinLab/oncoGAN/tree/main/LICENSE) ![training](https://badgen.net/badge/training/v0.2/blue) ![simulating](https://badgen.net/badge/simulating/v0.2.1/blue) ![fasta2bam](https://badgen.net/badge/fasta2bam/v0.1/blue)
 [![zenodo](https://img.shields.io/badge/docs-zenodo-green)](https://zenodo.org/records/13946726) [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13946726.svg)](https://doi.org/10.5281/zenodo.13946726) [![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm-dark.svg)](https://huggingface.co/anderdnavarro/OncoGAN) [![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm-dark.svg)](https://huggingface.co/collections/anderdnavarro/oncogan-67110940dcbafe5f1aa2d524)

# OncoGAN

A pipeline that accurately simulates high quality publicly cancer genomes (VCFs, CNAs and SVs) for eight different tumor types: Breast-AdenoCa, CNS-PiloAstro, Eso-AdenoCa, Kidney-RCC, Liver-HCC, Lymph-CLL, Panc-Endocrine and Prost-AdenoCa. OncoGAN offers a solution to current challenges in data accessibility and privacy while also serving as a powerful tool for enhancing algorithm development and benchmarking.

In addition to this pipeline, we have released 200 simulated VCFs for each of the eight studied tumor types, and that are availbale on [HuggingFace](https://huggingface.co/datasets/anderdnavarro/OncoGAN-syntheticVCFs) and [Zotero](https://zenodo.org/records/13946726).

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
    - [Baseline command](#baseline-command)
    - [CTAB-GAN+ models](#ctab-gan-models)
    - [CTGAN/TVAE models](#ctgantvae-models)
4. [DeepTumour](#deeptumour)
5. [Create tumor BAMs](#create-tumor-bams)
    - [Preparation of a tumor FASTA genome](#preparation-of-a-tumor-fasta-genome)
    - [InSilicoSeq](#insilicoseq)
    - [BAMsurgeon](#bamsurgeon)

## Installation

We have created three docker images with all dependencies installed as there are version incompatibility issues between the different modules: 

- Training -> Environment and scripts used to train OncoGAN models (CUDA)
- Simulating -> Pipeline for synthetic tumor simulation (CPU only)
- DeepTumour -> Algorithm developed to detect the tumor type of origin based o somatic mutations ([Ref](https://www.nature.com/articles/s41467-019-13825-8))
- fasta2bam -> Module to generate FASTQ/BAM files using OncoGAN's output

However, due to the size of the models, they couldn’t be stored in the Docker images and need to be downloaded separately (*see [Download models](#Download-models) section below*).

### Docker

If you don't have docker already installed in your system, please follow these [instructions](https://docs.docker.com/get-docker/).

```bash
# Training
docker pull oicr/oncogan:training_v0.2

# Simulating
docker pull oicr/oncogan:simulating_v0.2.1

# DeepTumour
docker pull ghcr.io/lincolnsteinlab/deeptumour:3.0

# fasta2bam
docker pull oicr/oncogan:fasta2bam_v0.1
```

### Singularity

If you don't have singularity already installed in your system, please follow these [instructions](https://apptainer.org/admin-docs/master/installation.html).

```bash
# Training
singularity pull docker://oicr/oncogan:training_v0.2

# Simulating
singularity pull docker://oicr/oncogan:simulating_v0.2.1

# DeepTumour
singularity pull docker://ghcr.io/lincolnsteinlab/deeptumour:3.0

# fasta2bam
singularity pull docker://oicr/oncogan:fasta2bam_v0.1
```

### Download models

OncoGAN trained models for the eight tumor types and DeepTumour models can be found on [HuggingFace](https://huggingface.co/anderdnavarro/OncoGAN) and [Zotero](https://zenodo.org/records/13946726).

## Generate synthetic VCFs

OncoGAN needs two external inputs to simulate new samples:

1. The directory with OncoGAN models downloaded previously
2. **hg19 fasta** reference genome without the *chr* prefix 

The output is a VCF file (mutations), two TSV files (CNAs and SVs) and a PNG (CNA+SV plot) per donor. Since the PCAWG dataset used for training refers to the hg19 version of the genome, the new mutations are also aligned to that version. The integrated `LiftOver` version can be used to swicht to hg38.

### Tumors with real profiles

```bash
# Docker command
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -v /PATH_TO_HG19_DIR/:/reference \
           -v /PATH_TO_ONCOGAN_MODELS/:/oncoGAN/trained_models \
           -it oicr/oncogan:simulating_v0.2.1 \
           vcfGANerator -n 1 --tumor Breast-AdenoCa -r /reference/hs37d5.fa [--hg38]

# Singularity command
singularity exec -H ${pwd}:/home \
            -B /PATH_TO_HG19_DIR/:/reference \
            -B /PATH_TO_ONCOGAN_MODELS/:/oncoGAN/trained_models \
            /PATH_TO/oncogan_simulating_v0.2.1.sif launcher.py \
            vcfGANerator -n 1 --tumor Breast-AdenoCa -r /reference/hs37d5.fa [--hg38]
```

The options for the `vcfGANerator` function are:

```bash
vcfGANerator --help

# Command to simulate mutations (VCF), CNAs and SVs for different tumor types using a GAN model

# Options:
#   -@, --cpus INTEGER      Number of CPUs to use  [default: 1]
#   --tumor TEXT            Tumor type to be simulated. Run 'availTumors'
#                           subcommand to check the list of available tumors that
#                           can be simulated  [required]
#   -n, --nCases INTEGER    Number of cases to simulate  [default: 1]
#   --NinT FLOAT            Normal in Tumor contamination to be taken into account when 
#                           adjusting VAF for CNA-SV events (e.g. 0.20 = 20%) [default: 0.0]
#   -r, --refGenome PATH    hg19 reference genome in fasta format  [required]
#   --prefix TEXT           Prefix to name the output. If not, '--tumor' option is
#                           used as prefix
#   --outDir DIRECTORY      Directory where save the simulations. Default is the
#                           current directory
#   --hg38                  Transform the mutations to hg38
#   --mut / --no-mut        Simulate mutations  [default: mut]
#   --CNA-SV / --no-CNA-SV  Simulate CNA and SV events  [default: CNA-SV]
#   --plots / --no-plots    Save plots  [default: plots]
#   --version               Show the version and exit
#   --help                  Show this message and exit
```

### Tumors with custom profiles

To generate tumors with custom profiles, users can use the [template](simulating/template_custom_simulation.csv), which contains a list of possible mutation types and signatures to simulate. If no CNA-SV are required, the `cna-sv profile` can be set to `-`.

```bash
# Docker command
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -v /PATH_TO_HG19_DIR/:/reference \
           -v /PATH_TO_ONCOGAN_MODELS/:/oncoGAN/trained_models \
           -it oicr/oncogan:simulating_v0.2.1 \
           vcfGANerator-custom --template /home/template_custom_simulation.csv -r /reference/hs37d5.fa [--hg38]

# Singularity command
singularity exec -H ${pwd}:/home \
            -B /PATH_TO_HG19_DIR/:/reference \
            -B /PATH_TO_ONCOGAN_MODELS/:/oncoGAN/trained_models \
            /PATH_TO/oncogan_simulating_v0.2.1.sif launcher.py \
            vcfGANerator-custom --template /home/template_custom_simulation.csv -r /reference/hs37d5.fa [--hg38]
```

The options for the `vcfGANerator-custom` function are:

```bash
vcfGANerator-custom --help

# Command to simulate mutations (VCF), CNAs and SVs for personalized tumors using a GAN model

# Options:
#   -@, --cpus INTEGER      Number of CPUs to use  [default: 1]
#   --template PATH         File in CSV format with the number of each type of
#                           mutation to simulate for each donor (template
#                           available on GitHub)  [required]
#   -r, --refGenome PATH    hg19 reference genome in fasta format  [required]
#   --outDir DIRECTORY      Directory where save the simulations. Default is the
#                           current directory
#   --hg38                  Transform the mutations to hg38
#   --CNA-SV / --no-CNA-SV  Simulate CNA and SV events  [default: CNA-SV]
#   --plots / --no-plots    Save plots  [default: plots]
#   --version               Show the version and exit
#   --help                  Show this message and exit
```

Among all the options offered by docker (`docker run --help`), we recommend:

- `--rm`: Automatically remove the container when it exits.
- `-u, --user`: Specify the user ID and its group ID. It's useful to not run the pipeline as root.
- `-v, --volume`: Mount local volumes in the container.
  - With the option `-v $(pwd):/home/`, OncoGAN results will be in your current directory.
- `-i, --interactive`: Keep STDIN open even if not attached.
- `-t, --tty`: Allocate a pseudo-TTY. When combined with `-i` it allows you to connect your terminal with the container terminal.

For singularity, the `-H` and `-B` options are analogous to `-v` docker option.

### More options 

List of available tumors:

```bash
docker run --rm -it oicr/oncogan:simulating_v0.2.1 availTumors

# or 

singularity exec /PATH_TO/oncogan_simulating_v0.2.1.sif launcher.py availTumors

# This is the list of available tumor types that can be simulated using OncoGAN:
# Breast-AdenoCa          CNS-PiloAstro           Eso-AdenoCa             Kidney-RCC              
# Liver-HCC               Lymph-CLL               Panc-Endocrine          Prost-AdenoCA
```

## Train new models

Files used to train OncoGAN models can be found [HuggingFace](https://huggingface.co/datasets/anderdnavarro/OncoGAN-training_files) and [Zotero](https://zenodo.org/records/13946726). The directory containing these files or your custom training files need to be mounted into the docker/singularity container.

We used two different training approaches: 

- [CTAB-GAN+](https://github.com/Team-TUD/CTAB-GAN-Plus) -> To train *donor characteristics*, *drivers*, *mutational signatures* and *CNA and SV features* 
- [CTGAN + TVAE](https://docs.sdv.dev/sdv) -> To train *genomic positions*

### Baseline command

The baseline command to run the docker/singularity container is the same for all the different training modalities:

```bash
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -v /PATH_TO_TRAINING_FILES/:/inputs \
           -p 8890:8890 \ #Only for CTGAN/TVAE models
           -it oicr/oncogan:training_v0.2 --help

# or

singularity exec -H ${pwd}:/home \
            -B /PATH_TO_TRAINING_FILES/:/inputs \
            /PATH_TO/oncogan_training_v0.2.sif launcher.py --help

# Options:
#   --help  Show this message and exit.

# Commands:
#   jupyter              Launches a jupyter lab instance
#   testHyperparameters  Test hyperparameters for counts/drivers CTABGAN...
#   trainCNA             Train a CTABGAN model for CNAs
#   trainCounts          Train a CTABGAN model for donor characteristics
#   trainDrivers         Train a CTABGAN model for driver features
#   trainMutations       Train a CTABGAN model for mutations
#   trainSV              Train a CTABGAN model for SVs
#   useModel             Use a CTABGAN model to generate synthetic data
```

### CTAB-GAN+ models

#### `trainCounts` / `trainDrivers` / `trainMutations` / `trainCNA` / `trainSV` 

Commands to train *donor characteristics*, *drivers*, *mutational signatures* and *CNA and SV features* models, respectively. The output will be a model and a simulated file with the same format and shape as the training file. All scripts have a very similar usage:

```bash
# Baseline command +
trainCounts --help

# Options:
#   --csv PATH                  CSV file with the counts used to train the model
#                               [required]
#   --prefix TEXT               Prefix to use to save the model  [required]
#   --outdir DIRECTORY          Directory where save the model  [default: /home]
#   --epochs INTEGER            Number of epochs  [default: 100]
#   --batch_size INTEGER        Batch size  [default: 20]
#   --test_ratio FLOAT          Test ratio  [default: 0.3]
#   --lr FLOAT                  Learning rate  [default: 0.0002]
#   --categorical_columns TEXT  Categorical columns. Comma separated with no
#                               space (e.g. x,y,z)
#   --log_columns TEXT          Log columns. Comma separated with no space (e.g.
#                               x,y,z)
#   --integer_columns TEXT      Integer columns. Comma separated with no space
#                               (e.g. x,y,z)
#   --no-tqdm                   Disable tqdm progress bar
#   --help                      Show this message and exit
```

#### `testHyperparameters`

Command to automatically test a combination of hyperparameters for *donor characteristics (counts)* and *drivers* models. The output will be a model and a simulated file for each hyperparameter combination.

```bash
# Baseline command +
testHyperparameters --help

# Test hyperparameters for counts/drivers CTABGAN models

# Options:
#   --cpu INTEGER                   Number of CPUs to use  [default: 1]
#   --function [counts|drivers]     Function to test hyperparameters for [required]
#   --csv PATH                      CSV file with the counts used to train the model  [required]
#   --prefix TEXT                   Prefix to use to save the model  [required]
#   --outdir DIRECTORY              Directory where save the model  [default: /home]
#   --epochs <INTEGER INTEGER INTEGER>...
#                                   A list with an epoch range: start stop step
#   --batch_size <INTEGER INTEGER INTEGER>...
#                                   A list with a batch_size range: start stop step
#   --lr <FLOAT FLOAT FLOAT>...     A list with a learning rate range: start stop step
#   --categorical_columns TEXT      Categorical columns. Comma separated with no
#                                   space (e.g. x,y,z)
#   --log_columns TEXT              Log columns. Comma separated with no space
#                                   (e.g. x,y,z)
#   --integer_columns TEXT          Integer columns. Comma separated with no
#                                   space (e.g. x,y,z)
#   --debug                         Greater verbosity for debugging purposes
#   --help                          Show this message and exit
```

An example of how to specify the range of the hyperparameters to test would be:

```bash
# Baseline command +
testHyperparameters --cpu 1 --function counts --csv /inputs/counts/Breast-AdenoCa_counts.csv --prefix Breast-AdenoCa --epochs 100 500 20 --batch_size 10 30 5 --lr 0.001 0.01 0.001 --integer_columns DEL,DNP,INS,TNP,SBS1,SBS2,SBS3,SBS5,SBS8,SBS13,SBS18
```

#### `useModel`

Training commands return the model and a simulated file with the same format and shape as the training file. To generate more simulated files to check the performance of the model you can run the `useModel` command:

```bash
# Baseline command +
useModel --help

# Use a driver/count CTABGAN model to generate synthetic data

# Options:
#   --model PATH        Model to use for the simulations (counts/drivers) [required]
#   --outdir DIRECTORY  Directory where save the simulations [default: /home]
#   --nDonors INTEGER   Number of donors to simulate for each simulation [default: 1]
#   --nFiles INTEGER    Number of files to simulate  [default: 1]
#   --help              Show this message and exit
```

### CTGAN/TVAE models

To train genomic position models using the CTGAN/TVAE approach we use a jupyter notebook inside the docker/singularity container. The output will be a model and a simulated file with the same format and shape as the training file.

The notebook with all the steps to train the models can be found [here](https://github.com/LincolnSteinLab/oncoGAN/blob/main/training/src/ctgan/position_simulation_training_example.ipynb).

```bash
# Baseline command +
jupyter --help

# Launches a jupyter lab instance

# Options:
#   --port INTEGER  Port to launch jupyter lab on  [default: 8890]
#   --help          Show this message and exit.
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
           ghcr.io/lincolnsteinlab/deeptumour:3.0 --help

# or

singularity exec \
            -B $(pwd):/WORKDIR \
            -B /PATH_TO_DEEPTUMOUR_MODEL//home/deeptumour/src/trained_models \
            -B /PATH_TO_HG19_DIR/:/reference \
            /PATH_TO/deeptumour_3.0.sif --help

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