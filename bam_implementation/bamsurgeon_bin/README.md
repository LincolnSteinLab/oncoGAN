# BAMsurgeon modifications

[BAMsurgeon](https://github.com/adamewing/bamsurgeon/tree/master) was installed using its conda version 1.4.1. From that specific version, the `addsv.py` script was modified to fix a bug trying to remove a BAM index that was never generated (line 799):

```python
# Original
if action == 'BIGDUP':
    bdup_left_bnd = min(region_1_start, region_2_start, region_1_end, region_2_end)
    bdup_right_bnd = max(region_1_start, region_2_start, region_1_end, region_2_end)
    prev_outbam_mutsfile = outbam_mutsfile
    outbam_mutsfile = add_donor_reads(args, mutid, outbam_mutsfile, chrom, bdup_left_bnd, bdup_right_bnd, float(svfrac))
    os.remove(prev_outbam_mutsfile)
    os.remove(prev_outbam_mutsfile + '.bai')

# Updated
if action == 'BIGDUP':
    bdup_left_bnd = min(region_1_start, region_2_start, region_1_end, region_2_end)
    bdup_right_bnd = max(region_1_start, region_2_start, region_1_end, region_2_end)
    prev_outbam_mutsfile = outbam_mutsfile
    outbam_mutsfile = add_donor_reads(args, mutid, outbam_mutsfile, chrom, bdup_left_bnd, bdup_right_bnd, float(svfrac))
    os.remove(prev_outbam_mutsfile)
    try:
        os.remove(prev_outbam_mutsfile + '.bai')
    except FileNotFoundError:
        pass
```