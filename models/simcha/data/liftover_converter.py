#!/usr/bin/env python3
"""
Convert hg19 coordinates to hg38 in SPICE data files using liftover.
Preserves the directory structure and all additional columns.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
from liftover import ChainFile

def initialize_chain_file() -> ChainFile:
    """Initialize the liftover chain file."""
    converter_path = '/.liftover/hg19ToHg38.over.chain.gz'
    if not os.path.exists(converter_path):
        raise FileNotFoundError(f"Chain file not found at {converter_path}")
    return ChainFile(converter_path)

def liftover_coordinate(converter: ChainFile, chrom: str, start: int, end: int) -> Optional[Tuple[str, int, int]]:
    """
    Convert a single coordinate pair from hg19 to hg38 using liftover.
    
    Args:
        converter: ChainFile instance for liftover conversion
        chrom: Chromosome name (should include 'chr' prefix)
        start: Start position (0-based)
        end: End position (0-based)
    
    Returns:
        Tuple of (new_chrom, new_start, new_end) or None if conversion fails
    """
    # Ensure chromosome has 'chr' prefix for liftover
    if not chrom.startswith('chr'):
        chrom = f'chr{chrom}'
    
    try:
        # convert_coordinate takes chrom as first arg and a position as second
        # Returns list of (new_chrom, new_start, strand) or None if failed
        # We need to convert the interval [start, end] by checking the start position
        result_start = converter.convert_coordinate(chrom, start)
        result_end = converter.convert_coordinate(chrom, end - 1)  # end-1 for 0-based inclusive
        
        if result_start and result_end:
            new_chrom_start, new_start_pos, strand_start = result_start[0]
            new_chrom_end, new_end_pos, strand_end = result_end[0]
            
            # Verify both coordinates map to the same chromosome and same strand
            if new_chrom_start == new_chrom_end and strand_start == strand_end:
                # Return the converted interval with +1 to end for proper 0-based coordinates
                return (new_chrom_start, new_start_pos, new_end_pos + 1)
        
        # Liftover failed for this coordinate
        return None
    except Exception as e:
        print(f"Error lifting over {chrom}:{start}-{end}: {e}", file=sys.stderr)
        return None

def convert_tsv_file(converter: ChainFile, input_path: Path, output_path: Path) -> None:
    """
    Convert a single TSV file from hg19 to hg38.
    
    Args:
        converter: ChainFile instance for liftover conversion
        input_path: Path to input hg19 TSV file
        output_path: Path to output hg38 TSV file
    """
    # Read the input file with full precision preservation
    df = pd.read_csv(input_path, sep='\t', float_precision='round_trip')
    
    # Store original columns for later
    original_columns = df.columns.tolist()
    
    # Check that required columns exist
    required_cols = ['chrom', 'start', 'end']
    if not all(col in original_columns for col in required_cols):
        raise ValueError(f"Input file {input_path} missing required columns: {required_cols}")
    
    # Convert coordinates
    converted_rows = []
    failed_count = 0
    
    for idx, row in df.iterrows():
        chrom = row['chrom']
        start = int(row['start'])
        end = int(row['end'])
        
        result = liftover_coordinate(converter, chrom, start, end)
        
        if result:
            new_chrom, new_start, new_end = result
            # Create new row with converted coordinates
            new_row = row.copy()
            new_row['chrom'] = new_chrom
            new_row['start'] = new_start
            new_row['end'] = new_end
            converted_rows.append(new_row)
        else:
            # Track failed conversions
            failed_count += 1
            # new_row = row.copy()
            # new_row['start'] = "NA"
            # new_row['end'] = "NA"
            # converted_rows.append(new_row)
    
    # Create output dataframe
    if converted_rows:
        output_df = pd.DataFrame(converted_rows)
        # Ensure column order matches original
        output_df = output_df[original_columns]
    else:
        output_df = pd.DataFrame(columns=original_columns)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to output file with full precision preservation
    output_df.to_csv(output_path, sep='\t', index=False, float_format='%.16g')
    
    print(f"Converted: {input_path.name}")
    print(f"  Rows: {len(df)} -> {len(output_df)} (failed: {failed_count})")
    print(f"  Output: {output_path}")

def main():
    """Main conversion pipeline."""
    # Paths
    hg19_base = Path('/workspace/models/simcha/data/hg19')
    hg38_base = Path('/workspace/models/simcha/data/hg38')
    
    # Validate input directory
    if not hg19_base.exists():
        print(f"Error: Input directory not found: {hg19_base}", file=sys.stderr)
        sys.exit(1)
    
    # Initialize liftover converter
    print("Initializing liftover converter...")
    converter = initialize_chain_file()
    print("✓ Liftover converter ready\n")
    
    # Find all spice_* directories
    spice_dirs = sorted([d for d in hg19_base.iterdir() if d.is_dir() and d.name.startswith('spice_')])
    
    if not spice_dirs:
        print(f"Warning: No spice_* directories found in {hg19_base}")
        sys.exit(0)
    
    print(f"Found {len(spice_dirs)} spice_* directories\n")
    
    total_files = 0
    failed_files = 0
    
    # Process each spice_* directory
    for spice_dir in spice_dirs:
        print(f"\nProcessing: {spice_dir.name}")
        print("=" * 60)
        
        # Find all TSV files in this directory
        tsv_files = sorted(spice_dir.glob('*.tsv'))
        
        if not tsv_files:
            print(f"  Warning: No TSV files found in {spice_dir}")
            continue
        
        # Process each TSV file
        for tsv_file in tsv_files:
            total_files += 1
            try:
                # Determine output path (same structure in hg38 directory)
                relative_path = tsv_file.relative_to(hg19_base)
                output_path = hg38_base / relative_path
                
                # Convert the file
                convert_tsv_file(converter, tsv_file, output_path)
                
            except Exception as e:
                print(f"  ERROR processing {tsv_file.name}: {e}", file=sys.stderr)
                failed_files += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Total files processed: {total_files}")
    print(f"Failed files: {failed_files}")
    print(f"Output directory: {hg38_base}")
    
    if failed_files > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
