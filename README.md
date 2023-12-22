# GenomeGAN

This repository contains the code to efficiently build and use three independent docker images:
- **genomegan:training.v0**: to train the genomeGAN models used for the simulation
- **genomegan:simulating.v0**: to simulate VCFs using the trained models. This only contains torch+cpu, which allows the building of a much lighter image 
- **genomegan:deeptumour**: to run DeepTumour on the simulated VCFs

## How to generate synthetic VCFs using genomeGAN

```bash
docker run --rm -u $(id -u):$(id -g) -v $(pwd):/home -v /home/adiaz-navarro/steinlab/databases/hg19:/reference -v /home/adiaz-navarro/steinlab/docker/genomeGAN/simulating/trained_models/:/genomeGAN/trained_models -it genomegan:simulating.v0 vcfGANerator -n 1 --tumor CNS-PiloAstro -r /reference/hs37d5.fa
```

## How to train counts or mutations using CTAB-GAN-Plus model

```bash
# Counts
docker run --rm -u $(id -u):$(id -g) -v $(pwd):/home -it genomegan:training.v0 trainCounts --csv /home/gan_mut_v7_3_CNS-PiloAstro_sig_counts.csv --prefix CNS-PiloAstro --epochs 230 --batch_size 15 --lr 0.0015

# Mutations
docker run --rm -u $(id -u):$(id -g) -v $(pwd):/home -it genomegan:training.v0 trainCounts --csv /home/gan_mut_v7_3_CNS-PiloAstro_sig_counts.csv --prefix CNS-PiloAstro --epochs 10000 --batch_size 200 --test_ratio 0.3 --lr 0.002
```

## How to train positions using CTGAN model

For the training of positions is better to use the jupyter notebook, since we have to check whether the first step is accurate enough to train the second step. To work with these jupyter notebooks is better to run a jupyter server using the docker environment.

```bash
docker run --rm -u $(id -u):$(id -g) -v $(pwd):/home -p 8890:8890 -it genomegan:training.v0 jupyter
```
