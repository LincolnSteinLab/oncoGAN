# GenomeGAN

This repository contains the code to efficiently build and use three independent docker images:
- **genomegan:training.v0**: to train the genomeGAN models used for the simulation
- **genomegan:simulating.v0**: to simulate VCFs using the trained models. This only contains torch+cpu, which allows the building of a much lighter image 
- **genomegan:deeptumour**: to run DeepTumour on the simulated VCFs

## How to generate synthetic VCFs using genomeGAN

```bash
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -v /home/adiaz-navarro/steinlab/databases/hg19:/reference \
           -v /home/adiaz-navarro/steinlab/docker/genomeGAN/simulating/trained_models/:/genomeGAN/trained_models \
           -it genomegan:simulating.v0 \
           vcfGANerator -n 1 --tumor CNS-PiloAstro -r /reference/hs37d5.fa

singularity exec -H ${pwd}:/home \
            -B /u/adiaz-navarro/adiaz/databases/hg19/:/reference \
            -B /u/adiaz-navarro/adiaz/projects/genome_simulator/mutations_distribution/gan/genomeGAN_simulations/trained_models/:/genomeGAN/trained_models \
            /u/adiaz-navarro/adiaz/venv/singularity/genomegan_simulating.sif launcher.py \
            vcfGANerator -n 1 --tumor CNS-PiloAstro -r /reference/hs37d5.fa
```

## How to train counts, mutations or drivers using CTAB-GAN-Plus model

```bash
# Counts
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -it genomegan:training.v0 \
           trainCounts --csv /home/gan_mut_v7_3_CNS-PiloAstro_sig_counts.csv --prefix CNS-PiloAstro --epochs 230 --batch_size 15 --lr 0.0015

singularity exec -H ${pwd}:/home \
            /u/adiaz-navarro/adiaz/venv/singularity/genomegan_training.sif launcher.py \
            trainCounts --csv /home/gan_mut_v7_3_CNS-PiloAstro_sig_counts.csv --prefix CNS-PiloAstro --epochs 230 --batch_size 15 --lr 0.0015

## Test trained counts
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -it genomegan:training.v0 \
           useModel --model /home/CNS-PiloAstro_counts_epoch220_batchsize10_lr0.007.pkl --nFiles 7 --nDonors 89

# Mutations
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -it genomegan:training.v0 \
           trainMutations --csv /home/gan_mut_v7_3_CNS-PiloAstro_sig_counts.csv --prefix CNS-PiloAstro --epochs 10000 --batch_size 200 --test_ratio 0.3 --lr 0.002

singularity exec -H ${pwd}:/home \
            /u/adiaz-navarro/adiaz/venv/singularity/genomegan_training.sif launcher.py \
            trainMutations --csv /home/gan_mut_v7_3_CNS-PiloAstro_sig_counts.csv --prefix CNS-PiloAstro --epochs 10000 --batch_size 200 --test_ratio 0.3 --lr 0.002

# Drivers
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -it genomegan:training.v0 \
           trainDrivers --csv /home/gan_drivers_CNS-PiloAstro.csv --prefix CNS-PiloAstro --epochs 230 --batch_size 15 --lr 0.0015

singularity exec -H ${pwd}:/home \
            /u/adiaz-navarro/adiaz/venv/singularity/genomegan_training.sif launcher.py \
            trainDrivers --csv /home/gan_drivers_CNS-PiloAstro.csv --prefix CNS-PiloAstro --epochs 230 --batch_size 15 --lr 0.0015

## Test trained drivers
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -it genomegan:training.v0 \
           useModel --model /home/CNS-PiloAstro_drivers_epoch100_batchsize15_lr0.005.pkl --nFiles 7 --nDonors 89

# Test hyperparameters
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -it genomegan:training.v0 \
           testHyperparameters --cpu 2 --function drivers --csv /home/gan_drivers_CNS-PiloAstro.csv --prefix CNS-PiloAstro --epochs 100 400 20 --batch_size 10 30 5 --lr 0.001 0.01 0.001 

singularity exec -H ${pwd}:/home \
            /u/adiaz-navarro/adiaz/venv/singularity/genomegan_training.sif launcher.py \
            testHyperparameters --cpu 2 --function drivers --csv /home/gan_drivers_CNS-PiloAstro.csv --prefix CNS-PiloAstro --epochs 100 400 20 --batch_size 10 30 5 --lr 0.001 0.01 0.001 
```

## How to train positions using CTGAN model

For the training of positions is better to use the jupyter notebook, since we have to check whether the first step is accurate enough to train the second step. To work with these jupyter notebooks is better to run a jupyter server using the docker environment.

```bash
docker run --rm -u $(id -u):$(id -g) \
           -p 8890:8890 \
           -v $(pwd):/home \
           -it genomegan:training.v0 jupyter

singularity exec -H ${pwd}:/home \
            -B /u/adiaz-navarro/adiaz/projects/genome_simulator/mutations_distribution/gan/files/positions/:/mnt/
            /u/adiaz-navarro/adiaz/venv/singularity/genomegan_training.sif jupyter-lab --no-browser --port 8890 --ip=`hostname` 
```

## How to run DeepTumour on the simulated VCFs

```bash
# Single VCF
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -v /home/adiaz-navarro/steinlab/docker/genomeGAN/deeptumour/trained_models:/DeepTumour/trained_models \
           -v /home/adiaz-navarro/steinlab/databases/hg19/:/reference \
           -it genomegan:deeptumour --vcfFile /home/CNS-PiloAstro_1.vcf --hg19 /reference/hs37d5.fa

# Multiple VCFs
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -v /home/adiaz-navarro/steinlab/docker/genomeGAN/deeptumour/trained_models:/DeepTumour/trained_models \
           -v /home/adiaz-navarro/steinlab/databases/hg19/:/reference \
           -it genomegan:deeptumour --vcfDir /home/vcf --hg19 /reference/hs37d5.fa

# hg38 VCF
docker run --rm -u $(id -u):$(id -g) \
           -v $(pwd):/home \
           -v /home/adiaz-navarro/steinlab/docker/genomeGAN/deeptumour/trained_models:/DeepTumour/trained_models \
           -v /home/adiaz-navarro/steinlab/databases/hg19/:/reference \
           -it genomegan:deeptumour --vcfFile /home/CNS-PiloAstro_1.vcf --hg19 /reference/hs37d5.fa --liftOver

# Singularity
singularity exec -H ${pwd}:/home \
            -B /u/adiaz-navarro/adiaz/projects/genome_simulator/mutations_distribution/gan/scripts/deeptumour/trained_models:/DeepTumour/trained_models
            -B /u/adiaz-navarro/adiaz/databases/hg19/:/reference \
            /u/adiaz-navarro/adiaz/venv/singularity/genomegan_deeptumour.sif DeepTumour.py \
            --vcfFile /home/CNS-PiloAstro_1.vcf --hg19 /reference/hs37d5.fa

```