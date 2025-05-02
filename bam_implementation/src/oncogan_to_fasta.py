import os
import click
import random
import itertools
import pandas as pd
import subprocess
from pyfaidx import Fasta

# General functions
def sort_by_int_chrom(chrom) -> pd.DataFrame:
    
    """
    Sort a dataframe using integer chromosomes
    """

    if chrom == 'X':
        return 23
    if chrom == 'Y':
        return 24
    else:
        return int(chrom)

def read_vcf(input:click.Path, dbSNP:bool = False) -> pd.DataFrame:

    """
    A function to open the VCF as a pd DataFrame
    """
    
    vcf:pd.DataFrame = pd.read_table(input, sep = "\t", comment = '#',
                        names = ['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'])
    vcf = vcf.sort_values(by=['chrom', 'pos'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)
    
    if dbSNP:
        vcf = vcf.drop(columns=['qual', 'filter', 'info'])
    else:
        vcf['snv_id'] = vcf['id'].apply(lambda x: x.split('_')[0])
        vcf['id'] = vcf['id'].apply(lambda x: x.split('_')[1])
        try: #cna vcf
            vcf[['af', 'ms', 'ta', 'al', 'cn']] = vcf['info'].str.extract(r'AF=([\d.]+);MS=([^;]+);TA=([^;]+);AL=([^;]+);CN=([^;]+)')
            vcf['af'] = vcf['af'].astype(float)
            vcf['ta'] = vcf['ta'].astype(int)
        except ValueError: #no-cna vcf
            vcf[['af', 'ms']] = vcf['info'].str.extract(r'AF=([\d.]+);MS=([^;]+)')
            vcf['af'] = vcf['af'].astype(float)
        vcf = vcf.drop(columns=['qual', 'filter', 'info', 'ms'])
    
    return vcf

def add_polymorphisms(genome:Fasta, dbsnp:pd.DataFrame) -> tuple:

    """
    Add polymorphisms to the reference genome
    """

    # Randomly assign haplotypes for each SNP
    dbsnp['allele'] = random.choices(['allele_1_minor', 'allele_2_major', 'homozygous'], k = dbsnp.shape[0])

    updated_genome:dict = {}
    updated_positions:pd.DataFrame = pd.DataFrame()
    for chrom in genome.keys():
        for allele in ['allele_1_minor', 'allele_2_major']:
            mov:int = 0
            error_muts:pd.DataFrame = pd.DataFrame() 
            updated_chrom:str = str(genome[chrom])

            chrom_dbsnp:pd.DataFrame = dbsnp[(dbsnp['chrom'].astype('str') == str(chrom)) & (dbsnp['allele'].isin([allele, 'homozygous']))]
            for _,mut in chrom_dbsnp.iterrows():
                position = mut['pos']+mov-1
                if ((len(mut['ref']) == 1) and (len(mut['alt']) == 1)): #SNP
                    if mut['ref'][0] != updated_chrom[position]:
                        error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                        continue
                    else:
                        updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+1:]
                        continue
                elif ((len(mut['ref']) == 2) and (len(mut['alt']) == 2)): #DNP
                    if mut['ref'] != updated_chrom[position:position+2]:
                        error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                        continue
                    else:
                        updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+2:]
                        continue
                elif ((len(mut['ref']) == 3) and (len(mut['alt']) == 3)): #TNP
                    if mut['ref'][0] != updated_chrom[position:position+3]:
                        error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                        continue
                    else:
                        updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+3:]
                        continue
                elif (len(mut['ref']) > 1): #DEL
                    if mut['ref'][0] != updated_chrom[position]:
                        error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                        continue
                    else:
                        updated_chrom = updated_chrom[:position+1] + updated_chrom[position+len(mut['ref']):]
                        mov -= len(mut['ref'])-1
                        updated_positions = pd.concat([updated_positions, pd.DataFrame(data={'chrom': [chrom], 'pos': [mut['pos']], 'allele': [allele], 'mov': [mov]})])
                        continue
                elif (len(mut['alt']) > 1): #INS
                    if mut['ref'][0] != updated_chrom[position]:
                        error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                        continue
                    else:
                        updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+1:]
                        mov += len(mut['alt'])-1
                        updated_positions = pd.concat([updated_positions, pd.DataFrame(data={'chrom': [chrom], 'pos': [mut['pos']], 'allele': [allele], 'mov': [mov]})])
                        continue

            updated_genome[f'{chrom}_{allele}'] = updated_chrom
            updated_positions = updated_positions.reset_index(drop=True)

    return(updated_genome, dbsnp, updated_positions)

def initialize_genome(genome:Fasta) -> tuple:

    """
    Initialize the genome with the two alleles
    """

    updated_genome:dict = {}
    updated_positions:pd.DataFrame = pd.DataFrame()
    for chrom in genome.keys():
        updated_genome[f'{chrom}_allele_2_major'] = str(genome[chrom])
        updated_genome[f'{chrom}_allele_1_minor'] = str(genome[chrom])
        updated_positions = pd.concat([updated_positions, pd.DataFrame(data={'chrom': [chrom], 'pos': [0], 'allele': ['allele_2_major'], 'mov': [0]})])
        updated_positions = pd.concat([updated_positions, pd.DataFrame(data={'chrom': [chrom], 'pos': [0], 'allele': ['allele_1_minor'], 'mov': [0]})])

    return(updated_genome, updated_positions)

def get_mov(info_df:pd.DataFrame, mut:pd.Series, allele) -> int:

    """
    Keep the trace of the length difference of the custom genome with respect to the reference genome
    """

    info_df_filtered:pd.DataFrame = info_df[(info_df['chrom'].astype(str) == str(mut['chrom'])) & (info_df['pos'] < mut['pos']) & (info_df['allele'] == allele)]
    if info_df_filtered.empty:
        return 0
    else:
        last_row:pd.Series = info_df_filtered.tail(1).iloc[0]
        return last_row['mov']

# No-CNA specific functions
def assign_copies_apply(row:pd.Series, combinations:dict, copies:list) -> str:
    
    """
    Assign to which chromosome copies each mutation should be introduced based on its allele frequency
    """

    chrom:str = row['chrom']
    af:float = row['af']
    hap:str = row['hap']
    
    if af < 0.6:
        comb:tuple = best_combination(combinations, af)
        assignment_str:str = ','.join([f"{chrom}_freq{freq}_hap{hap}" for freq in comb])
        return assignment_str
    else:
        ## Major haplotype
        major_assignment:list = [f"{chrom}_freq{freq}_hap{hap[0]}" for freq in copies]
        ## Minor haplotype
        comb_minor:tuple = best_combination(combinations, af-sum(copies))
        minor_assignment:list = [f"{chrom}_freq{freq}_hap{hap[1]}" for freq in comb_minor]
        assignment:list = major_assignment + minor_assignment
        assignment_str:str = ','.join(assignment)
        return assignment_str

def subset_mutations_apply(x:pd.Series, key:str) -> bool:

    """
    Filter mutations by their assigned chromosome/allele/frequency
    """

    x = x.split(',')
    if key in x:
        return True
    else:
        return False

def best_combination(combinations:dict, af:float) -> tuple:
    
    """
    Return the subset of copies whose sum is closest to the target allele frequency
    """
    
    best_key:int = min(combinations.keys(), key=lambda k: abs(k - af))
    return combinations[best_key]

def assign_allele_copies(vcf:pd.DataFrame) -> pd.DataFrame:

    """
    Assign to which chromosome copies each mutation should be introduced based on its allele frequency
    """

    vcf['hap'] = vcf['af'].apply(lambda af: random.choice(['AB', 'BA']) if af > 0.6 else random.choice(['A', 'B']))
    copies:list = [0.3, 0.15, 0.05]
    possible_combinations:dict = {}
    for r in range(1, len(copies) + 1):
        for subset in itertools.combinations(copies, r):
            s:float = sum(subset)
            if s not in possible_combinations:
                possible_combinations[s] = subset
    vcf['copy_assignment'] = vcf.apply(assign_copies_apply, combinations=possible_combinations, copies=copies, axis=1)

    return vcf

def calculate_chrom_genome_abundance(refGenome:click.Path) -> dict:
    
    """
    Calculate the abundance of each chromosome in the reference genome
    """

    # Load chromosome lengths
    genome:pd.DataFrame = pd.read_csv(f"{refGenome}.fai", sep='\t', header=None, usecols=[0, 1], names=['chrom', 'length'])
    genome['chrom'] = genome['chrom'].astype(str)

    # Compute total genome length
    genome_length:int = genome['length'].sum()

    # Calculate relative abundance
    genome['abundance'] = genome['length'] / genome_length

    # Convert to dictionary
    chrom_genome_abundance:dict = genome.set_index('chrom')['abundance'].to_dict()
    
    return chrom_genome_abundance

def add_mutations(genome:dict, mutations:pd.DataFrame, germ_info:pd.DataFrame, chrom_abundance:dict, outDir:click.Path, donor_id:str) -> None:

    """
    Add mutations to the custom reference genome
    """

    # Randomly assign haplotypes for each mutation
    mutations = assign_allele_copies(mutations)

    abundance_df:pd.DataFrame = pd.DataFrame(columns=['chrom', 'abundance'])
    for chrom_allele in genome.keys():
        chrom:str = str(chrom_allele.split('_')[0])
        allele:str = '_'.join(chrom_allele.split('_')[1:])
        chrom_germ_info:pd.DataFrame = germ_info[(germ_info['chrom'].astype(str) == chrom) & (germ_info['allele'] == allele)]


        final_genome_path:click.Path = os.path.join(outDir, f"{donor_id}_genome.fa")
        if os.path.exists(final_genome_path):
            os.remove(final_genome_path)
        with open(final_genome_path, 'a') as final_genome: 
            for freq in ['0.3', '0.15', '0.05']:
                chrom_freq_allele = f'{chrom}_freq{freq}_{allele}'

                ## Create an abundance entry for each chromosome
                abundance_df = pd.concat([abundance_df, pd.DataFrame(data={'chrom': [chrom_freq_allele], 'abundance': [chrom_abundance[chrom]*float(freq)]})])
                
                ## Subset mutations
                chrom_mutations:pd.DataFrame = mutations[(mutations['chrom'].astype('str') == str(chrom))]
                chrom_mutations['keep'] = mutations['copy_assignment'].apply(lambda x: subset_mutations_apply(x, key=chrom_freq_allele))
                chrom_mutations = chrom_mutations[chrom_mutations['keep']]

                if chrom_mutations.empty:
                    final_genome.write(f'>{chrom_freq_allele}\n{genome[chrom_allele]}\n')
                    continue

                mov:int = 0
                error_muts:pd.DataFrame = pd.DataFrame()
                updated_chrom:str = genome[chrom_allele]
                for _,mut in chrom_mutations.iterrows():
                    chrom_germ_mov:int = get_mov(chrom_germ_info, mut, allele)
                    position = mut['pos']+mov+chrom_germ_mov-1
                    if ((len(mut['ref']) == 1) and (len(mut['alt']) == 1)): #SNP
                        if mut['ref'][0] != updated_chrom[position]:
                            error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                            continue
                        else:
                            updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+1:]
                            continue
                    elif ((len(mut['ref']) == 2) and (len(mut['alt']) == 2)): #DNP
                        if mut['ref'] != updated_chrom[position:position+2]:
                            error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                            continue
                        else:
                            updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+2:]
                            continue
                    elif ((len(mut['ref']) == 3) and (len(mut['alt']) == 3)): #TNP
                        if mut['ref'][0] != updated_chrom[position:position+3]:
                            error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                            continue
                        else:
                            updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+3:]
                            continue
                    elif (len(mut['ref']) > 1): #DEL
                        if mut['ref'][0] != updated_chrom[position]:
                            error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                            continue
                        else:
                            updated_chrom = updated_chrom[:position+1] + updated_chrom[position+len(mut['ref']):]
                            mov -= len(mut['ref'])-1
                            continue
                    elif (len(mut['alt']) > 1): #INS
                        if mut['ref'][0] != updated_chrom[position]:
                            error_muts = pd.concat([error_muts, mut.to_frame().T], ignore_index=False)
                            continue
                        else:
                            updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+1:]
                            mov += len(mut['alt'])-1
                            continue
                final_genome.write(f'>{chrom_freq_allele}\n{updated_chrom}\n')
    
    # Be sure abundance sum is 1
    abundance_df['abundance'] = abundance_df['abundance'] / abundance_df['abundance'].sum().round(3)
    abundance_df.to_csv(os.path.join(outDir, f"{donor_id}_abundance.txt"), sep='\t', index=False, header=False)

# CNA specific functions
def update_next_movs(info_df:pd.DataFrame, chrom:str, allele:str, pos:int, mov2up:int) -> pd.DataFrame:
    
    """
    Update changes in the reference genome length
    """

    muts2up:pd.Index = info_df[(info_df['chrom'].astype(str) == str(chrom)) & (info_df['allele'] == allele) & (info_df['pos'].astype(int) > pos)].index
    info_df.loc[muts2up, 'mov'] += mov2up

    return(info_df)

def get_sv_mov(germ_info:pd.DataFrame, somatic_info:pd.DataFrame, sv_info:pd.DataFrame, chrom:str, start:int, end:int, allele:str) -> tuple:

    """
    Helper to extract the updated coordinates for SVs
    """

    germ_mov:int = get_mov(germ_info, pd.Series(data=[chrom, start], index=['chrom', 'pos']), allele)
    somatic_mov:int = get_mov(somatic_info, pd.Series(data=[chrom, start], index=['chrom', 'pos']), allele)
    sv_mov:int = get_mov(sv_info, pd.Series(data=[chrom, start], index=['chrom', 'pos']), allele)
    start:int = start+sv_mov+somatic_mov+germ_mov-1

    germ_mov:int = get_mov(germ_info, pd.Series(data=[chrom, end], index=['chrom', 'pos']), allele)
    somatic_mov:int = get_mov(somatic_info, pd.Series(data=[chrom, end], index=['chrom', 'pos']), allele)
    sv_mov:int = get_mov(sv_info, pd.Series(data=[chrom, end], index=['chrom', 'pos']), allele)
    end:int = end+sv_mov+somatic_mov+germ_mov-1

    return(start, end, sv_mov)

def add_mutations_cna(genome:dict, mutations:pd.DataFrame, germ_info:pd.DataFrame, events:click.Path, svs:click.Path, outDir:click.Path, donor_id:str) -> None:

    """
    Add mutations to the custom reference genome
    """

    # Open event and CNA files
    events_df:pd.DataFrame = pd.read_csv(events, delimiter='\t')
    sv_df:pd.DataFrame = pd.read_csv(svs, delimiter='\t')

    # Follow the order of events to introduce the mutations and CNAs
    somatic_info:pd.DataFrame = pd.DataFrame(columns=['chrom', 'pos', 'allele', 'mov'])
    for _,event in events_df.iterrows():
        if event['class'] == "MUT":
            try:
                mut:pd.Series = mutations[mutations['snv_id'] == event['event_id']].iloc[0]
            except IndexError:
                ## In case the mutation is located in a deleted allele
                continue

            ## Extract position modifiers
            if germ_info is None:
                germ_mov:int = 0
            else:
                germ_mov:int = get_mov(germ_info, mut, event['allele'])
            somatic_mov:int = get_mov(somatic_info, mut, event['allele'])
            position:int = mut['pos']+somatic_mov+germ_mov-1

            ## Add the mutations
            chrom_key:str = f"{mut['chrom']}_{event['allele']}"        
            updated_chrom:str = genome[chrom_key]
            if ((len(mut['ref']) == 1) and (len(mut['alt']) == 1)): #SNP
                if mut['ref'][0] != updated_chrom[position]:
                    continue
                else:
                    updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+1:]
                    genome[chrom_key] = updated_chrom
                    continue
            elif ((len(mut['ref']) == 2) and (len(mut['alt']) == 2)): #DNP
                if mut['ref'] != updated_chrom[position:position+2]:
                    continue
                else:
                    updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+2:]
                    genome[chrom_key] = updated_chrom
                    continue
            elif ((len(mut['ref']) == 3) and (len(mut['alt']) == 3)): #TNP
                if mut['ref'][0] != updated_chrom[position:position+3]:
                    continue
                else:
                    updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+3:]
                    genome[chrom_key] = updated_chrom
                    continue
            elif (len(mut['ref']) > 1): #DEL
                if mut['ref'][0] != updated_chrom[position]:
                    continue
                else:
                    updated_chrom = updated_chrom[:position+1] + updated_chrom[position+len(mut['ref']):]
                    genome[chrom_key] = updated_chrom
                    somatic_mov -= len(mut['ref'])-1
                    somatic_info = pd.concat([somatic_info, pd.DataFrame(data={'chrom': [mut['chrom']], 'pos': [mut['pos']], 'allele': [event['allele']], 'mov': [somatic_mov]})])
                    somatic_info = somatic_info.sort_values(by=['chrom', 'pos'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)
                    somatic_info = update_next_movs(somatic_info, mut['chrom'], event['allele'], mut['pos'], -(len(mut['ref'])-1))
                    continue
            elif (len(mut['alt']) > 1): #INS
                if mut['ref'][0] != updated_chrom[position]:
                    continue
                else:
                    updated_chrom = updated_chrom[:position] + mut['alt'] + updated_chrom[position+1:]
                    genome[chrom_key] = updated_chrom
                    somatic_mov += len(mut['alt'])-1
                    somatic_info = pd.concat([somatic_info, pd.DataFrame(data={'chrom': [mut['chrom']], 'pos': [mut['pos']], 'allele': [event['allele']], 'mov': [somatic_mov]})])
                    somatic_info = somatic_info.sort_values(by=['chrom', 'pos'], key=lambda col: col.map(sort_by_int_chrom)).reset_index(drop=True)
                    somatic_info = update_next_movs(somatic_info, mut['chrom'], event['allele'], mut['pos'], len(mut['alt'])-1)
                    continue
        elif event['class'] == "DEL":
            ## Deletions are processed later
            continue
        elif event['class'] == "DUP":
            ## Create a new allele for the duplication
            chrom:str = str(event['cna_id'].split('_')[0].replace('x', ''))
            chrom_key:str = f"{chrom}_{event['from_allele']}" #parent allele
            dup_chrom_key:str = f"{chrom}_{event['to_allele']}" #new allele
            genome[dup_chrom_key] = genome[chrom_key]

            ## Update position modifiers for the new allele
            ### Germline
            tmp_germ:pd.DataFrame = germ_info[(germ_info['chrom'].astype(str) == chrom) & (germ_info['allele'] == event['from_allele'])]
            tmp_germ.loc[:,'allele'] = event['to_allele']
            germ_info = pd.concat([germ_info, tmp_germ]).reset_index(drop=True)
            ### Somatic
            tmp_som:pd.DataFrame = somatic_info[(somatic_info['chrom'].astype(str) == chrom) & (somatic_info['allele'] == event['from_allele'])]
            tmp_som.loc[:,'allele'] = event['to_allele']
            somatic_info = pd.concat([somatic_info, tmp_som]).reset_index(drop=True)

    # Introduce the CNA into the reference genome
    sv_info:pd.DataFrame = pd.DataFrame(columns=list(germ_info.columns))
    for _,sv in sv_df.iterrows():
        chrom:str = str(sv['chrom1'])
        start:int = sv['start1']
        end:int = sv['end2']
        sv_event:pd.Series = events_df[events_df['event_id'] == sv['sv_id']].iloc[0]
        parent_allele:str = f'{chrom}_allele_2_major' if 'major' in sv_event['from_allele'] else f'{chrom}_allele_1_minor'

        ## Get coordinates
        parent_start, parent_end, parent_mov = get_sv_mov(germ_info, somatic_info, sv_info, chrom, start, end, parent_allele)
        event_start, event_end, event_mov = get_sv_mov(germ_info, somatic_info, sv_info, chrom, start, end, event['to_allele'])

        if sv['svclass'] == "DUP":
            genome[parent_allele] = genome[parent_allele][:parent_end] + genome[f"{chrom}_{sv_event['to_allele']}"][event_start:event_end] + genome[parent_allele][parent_end:]
            parent_mov += end-start
            sv_info = pd.concat([sv_info, pd.DataFrame(data={'chrom': [chrom], 'pos': [end], 'allele': [parent_allele], 'mov': [parent_mov]})])
        elif sv['svclass'] == "DEL":
            genome[parent_allele] = genome[parent_allele][:parent_start] + genome[parent_allele][parent_end:]
            parent_mov -= end-start
            sv_info = pd.concat([sv_info, pd.DataFrame(data={'chrom': [chrom], 'pos': [end], 'allele': [parent_allele], 'mov': [parent_mov]})])

    # Write the new genome
    whole_genome:str = ''
    for chrom in genome.keys():
        if ('allele_2_major' in chrom) or ('allele_1_minor' in chrom):
            whole_genome += genome[chrom] + 'N'*1000
    custom_genome_path:click.Path = os.path.join(outDir, f"{donor_id}_genome.fa")
    with open(custom_genome_path, 'w') as custom_genome:
        custom_genome.write(f'>custom_genome\n{whole_genome}\n')

# CLI
@click.command(name="OncoGAN-to-FASTA")
@click.option("-i", "--input",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="VCF file containing the mutations simulated by OncoGAN")
@click.option("-r", "--reference_genome", "reference_genome",
              type=click.Path(exists=True, file_okay=True),
              required=True,
              help="Reference genome in FASTA format")
@click.option("--events",
              type=click.Path(exists=True, file_okay=True),
              required=False,
              default=None,
              show_default=True,
              help="TSV file containing the order of the events simulated by OncoGAN (together with --sv)[optional]")
@click.option("--sv",
              type=click.Path(exists=True, file_okay=True),
              required=False,
              default=None,
              show_default=True,
              help="TSV file containing the structural variants simulated by OncoGAN (together with --events) [optional]")
@click.option("--dbSNP", "dbSNP",
              type=click.Path(exists=True, file_okay=True),
              required=False,
              default=None,
              help="VCF file containing the germline mutations to be added to the reference genome [optional]")
@click.option("-o", "--out_dir",
              type=click.Path(exists=False, file_okay=False),
              default=os.getcwd(),
              show_default=False,
              help="Directory where save the FASTA genome. Default is the current directory")
def oncogan_to_fasta(input, reference_genome, events, sv, dbSNP, out_dir):

    """
    Create a custom FASTA genome containing the mutations simulated by OncoGAN
    """

    # Check options
    if (events is None and sv is not None) or (events is not None and sv is None):
        raise click.BadParameter("Both --events and --sv options must be provided to simulate CNAs.")
    
    # Check if the genome is indexed
    if not os.path.exists(reference_genome + ".fai"):
        print(f"Indexing {os.basename(reference_genome)} with samtools faidx")
        cmd:list = ["samtools", "faidx", reference_genome]
        subprocess.run(cmd, check=True)

    # Load reference genome
    genome:Fasta = Fasta(reference_genome)

    # Load the VCF
    vcf:pd.DataFrame = read_vcf(input)
    donor_id:str = vcf['id'][0]

    # Create directories
    os.makedirs(out_dir, exist_ok=True)
    
    # dbSNP
    if dbSNP is not None:
        ## Load dbSNP
        dbsnp_vcf:pd.DataFrame = read_vcf(dbSNP, dbSNP=True)

        ## Add dbSNP polymorphisms
        genome, dbsnp_vcf, updated_positions = add_polymorphisms(genome, dbsnp_vcf)
        dbsnp_vcf.to_csv(os.path.join(out_dir, f"{donor_id}_{os.path.basename(dbSNP).replace('.vcf', '_with_alleles.tsv')}"), sep='\t', index=False)
    else:
        genome, updated_positions = initialize_genome(genome)

    # Add mutations
    if events is None and sv is None:
        # Calculate genome abundance
        chrom_abundance:dict = calculate_chrom_genome_abundance(reference_genome)
        # Add OncoGAN mutations
        add_mutations(genome, vcf, updated_positions, chrom_abundance, out_dir, donor_id)
    else:
        # Add OncoGAN mutations and CNAs
        add_mutations_cna(genome, vcf, updated_positions, events, sv, out_dir, donor_id)
