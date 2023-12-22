# GenomeGAN

This repository contains the code to efficiently build and use three independent docker images:
- **genomegan:training.v0**: to train the genomeGAN models used for the simulation
- **genomegan:simulating.v0**: to simulate VCFs using the trained models. This only contains torch+cpu, which allows the building of a much lighter image 
- **genomegan:deeptumour**: to run DeepTumour on the simulated VCFs

## How to generate synthetic VCFs using genomeGAN

```bash
docker run --rm -u $(id -u):$(id -g) -v $(pwd):/home -v /home/adiaz-navarro/steinlab/databases/hg19:/reference -v /home/adiaz-navarro/steinlab/docker/genomeGAN/simulating/trained_models/:/genomeGAN/trained_models -it genomegan:simulating.v0 vcfGANerator -n 1 --tumor CNS-PiloAstro -r /reference/hs37d5.fa
``````