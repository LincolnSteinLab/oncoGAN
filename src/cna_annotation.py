import pickle
import re
import random
import pandas as pd
import numpy as np

def norm_chr(chrom):
    
    """
    Helper function to standardize chromosome names
    """
    
    return str(chrom).lower().replace('chr', '')

class Segment:
    def __init__(self, seg_id:int, name:str, hap:str, chrom:str, start:int, end:int, cna_id:int=0):
        self.seg_id:int = seg_id
        self.name:str = name
        self.hap:str = hap
        self.chrom:str = norm_chr(chrom)
        self.start:int = start
        self.end:int = end
        self.cna_id:int = cna_id
        self.mutations:dict = {}

    def contains(self, chrom:str, pos:int) -> bool:

        """
        Function to check if a given genomic position (mutation) falls within this segment
        """

        return self.chrom == norm_chr(chrom) and self.start <= pos <= self.end
    
    def extract(self, chrom:str, start:int, end:int) -> dict:
        
        """
        Function to extract mutations from this segment
        """
        
        q_chr:str = norm_chr(chrom)
        extracted:dict = {}
        for mut_id, mut in self.mutations.items():
            if mut['chrom'] == q_chr and start <= mut['pos'] <= end:
                extracted[mut_id] = mut.copy() 
        return extracted

class TumorGenome:
    def __init__(self, fai_path:str):
        self.segments_dict:dict = {}
        self.next_seg_id:int = 0

        with open(fai_path, 'r') as fai:
            for line in fai:
                cols:list = line.strip().split('\t')
                chrom:str = str(cols[0])
                length:int = int(cols[1])
                
                for hap in ['H1', 'H2']:
                    seg_name:str = f"{hap}>chr{chrom}[0:{length})_C1"
                    seg = Segment(
                        seg_id=self.next_seg_id,
                        name=seg_name,
                        hap=hap,
                        chrom=chrom,
                        start=0,
                        end=length
                    )
                    self.segments_dict[seg_name] = seg
                    self.next_seg_id += 1

    def parse_regions(self, regions_str:str) -> list[dict]:

        """
        Function to extract information (hap, chrom, start, end) from region string
        """
        
        if pd.isna(regions_str) or str(regions_str).strip() in ['[]', '']:
            return []
        
        pattern:str = r'(H[12])>chr([A-Za-z0-9_]+)\[(\d+):(\d+)\)'
        matches:list = re.findall(pattern, str(regions_str))
        
        return [{
            'hap': hap,
            'chrom': chrom,
            'start': int(start),
            'end': int(end)
        } for hap, chrom, start, end in matches]

    def find_overlapping_segments(self, hap:str, chrom:str, start:int, end:int) -> list[str]:

        """
        Function to find overlapping segments (same haplotype and chromosome)
        """
        
        q_chr:str = norm_chr(chrom)
        matches:list = []
        for seg_name, seg in self.segments_dict.items():
            if seg.hap != hap or seg.chrom != q_chr:
                continue
            if (seg.start < end) and (seg.end > start):
                matches.append(seg_name)
        return matches

    def apply_cna(self, row:pd.Series):

        """
        Function to remodelate segments based on copy number alteration events. It also transfers/removes mutations present in new/removed segments
        """
        
        # Extract event information
        event_type:str = row['event_type']
        
        r_lost_raw:str = row.get('regions_lost', '')
        r_gained_raw:str = row.get('regions_gained', '')
        
        regions_lost:list = [r.strip() for r in str(r_lost_raw).lstrip('[').rstrip(']').split(',') if r.strip()]
        regions_gained:list = [r.strip() for r in str(r_gained_raw).lstrip('[').rstrip(']').split(',') if r.strip()]
        regions_gained_parsed:list = self.parse_regions(r_gained_raw)

        # Create new segments
        to_rm:dict = {}
        if regions_gained: 
            for idx, region in enumerate(regions_gained):
                region_info:dict = regions_gained_parsed[idx]

                c_matching:list = [k for k in self.segments_dict.keys() if k.startswith(region)]
                next_copy:int = len(c_matching) + 1 if c_matching else 1
                new_seg_name:str = f"{region}_C{next_copy}"
                new_seg = Segment(
                    seg_id=self.next_seg_id, 
                    name=new_seg_name, 
                    hap=region_info['hap'],
                    chrom=region_info['chrom'],
                    start=region_info['start'],
                    end=region_info['end'],
                    cna_id=row['event_id']
                )
                self.segments_dict[new_seg_name] = new_seg
                self.next_seg_id += 1

                # If no regions were lost, inherit from matching/overlapping parents
                if not regions_lost:
                    intervals_to_extract:list[tuple[Segment, int, int]] = []
                    
                    # Scenario A: Exact match found
                    if c_matching:
                        chosen_parent:Segment = self.segments_dict[random.choice(c_matching)]
                        intervals_to_extract.append((
                            chosen_parent, 
                            region_info['start'], 
                            region_info['end']
                        ))
                    
                    # Scenario B: Need to stitch / choose from overlapping segments
                    else: 
                        c_start:int = region_info['start']
                        c_end:int = region_info['end']
                        
                        parent_names:list = self.find_overlapping_segments(
                            hap=region_info['hap'], chrom=region_info['chrom'],
                            start=c_start, end=c_end
                        )
                        candidate_parents:list = [self.segments_dict[n] for n in parent_names]
                        
                        # Cover segments to transfer non-overlapping mutations
                        curr:int = c_start
                        while curr < c_end:
                            ## Find all candidate segments that cover our current position
                            covering:list = [p for p in candidate_parents if p.start <= curr < p.end]
                            
                            ## If there's a gap in the graph, jump to the next available parent
                            if not covering:
                                future:list = [p for p in candidate_parents if p.start > curr]
                                if not future: break
                                curr = min(p.start for p in future)
                                continue
                                
                            ## Randomly choose one parent
                            chosen_parent:Segment = random.choice(covering)
                            
                            ## Check how far we can walk with this parent
                            chunk_end:int = min(c_end, chosen_parent.end)
                            
                            ## Save this specific chunk for extraction
                            intervals_to_extract.append((chosen_parent, curr, chunk_end))
                            
                            ## Move forward to where this parent ends
                            curr = chunk_end
                    
                    # Apply the calculated intervals
                    child_seg:Segment = self.segments_dict[new_seg_name]
                    for parent_seg, ext_start, ext_end in intervals_to_extract:
                        extracted:dict = parent_seg.extract(child_seg.chrom, ext_start, ext_end)
                        for mut_id, mut in extracted.items():
                            mut['allele'] = child_seg.name
                            mut['cn'] = child_seg.cna_id
                            child_seg.mutations[mut_id] = mut
        
        # Transfer parent mutations to new segments if regions were also lost
        if regions_gained and regions_lost:
            for region_l in regions_lost:
                c_matching_l:list = [k for k in self.segments_dict.keys() if k.startswith(region_l)]
                if not c_matching_l: 
                    continue
                
                region_l_ch:str = random.choice(c_matching_l)
                to_rm[region_l] = region_l_ch
                parent_seg:Segment = self.segments_dict[region_l_ch]
                
                for region_g in regions_gained:
                    c_matching_g:list = [k for k in self.segments_dict.keys() if k.startswith(region_g)]
                    if not c_matching_g: 
                        continue
                        
                    region_g_ch:str = random.choice(c_matching_g)
                    child_seg:Segment = self.segments_dict[region_g_ch]
                    inherited_mutations:dict = parent_seg.extract(child_seg.chrom, child_seg.start, child_seg.end)
                    
                    for mut_id, mut in inherited_mutations.items():
                        if event_type in ['InternalDuplication', 'CentromereBoundDuplication']:
                            mut['allele'] = child_seg.name
                            mut['cn'] = child_seg.cna_id
                        child_seg.mutations[mut_id] = mut
        
        # Remove lost segments
        if regions_lost:
            for region in regions_lost:
                region_to_rm:str = to_rm.get(region)
                if not region_to_rm:
                    c_matching:list = [k for k in self.segments_dict.keys() if k.startswith(region)]
                    if c_matching:
                        region_to_rm:str = random.choice(c_matching)
                if region_to_rm:
                    self.segments_dict.pop(region_to_rm, None)

    def assign_mutation(self, mut_id:int, info):

        """
        Function to assign a mutation to a specific segment
        """
        
        chrom, pos = info['#CHROM'], int(info['POS'])
        
        candidates:list = [seg for seg in self.segments_dict.values() if seg.contains(chrom, pos)]
        if candidates:
            chosen_seg:Segment = random.choice(candidates)
            chosen_seg.mutations[mut_id] = {
                'id': mut_id, 
                'chrom': norm_chr(chrom), 
                'pos': pos,
                'allele': chosen_seg.name,
                'cn': chosen_seg.cna_id
            }
