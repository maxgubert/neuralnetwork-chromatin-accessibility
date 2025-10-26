import pyBigWig
import py2bit
import argparse
import pandas as pd
import numpy as np


# Arguments
parser = argparse.ArgumentParser(description="Preprocessing")

parser.add_argument('--track', dest='track', type = str, required = True)
parser.add_argument('--chrom', dest='chrom', type = str, required = True)
parser.add_argument('--window_size', dest='window_size', type=int)

args = parser.parse_args()


# Open reference genome and track
reference = py2bit.open("../../data/hg38.2bit")
track = pyBigWig.open(args.track)


# Parameters
chrom = args.chrom
len_chr = reference.chroms(chrom)
window_size = args.window_size
n_windows = len_chr // window_size
flank_size = window_size // 2

n_windows = 0
n_windows_dropped_N = 0
n_windows_dropped_nan = 0

outfile_seq = []
outfile_values = []


# Functions
def mapping(seq, start, end):
    out = [str(chrom), str(start), str(end)]
    for x in seq:
        A, C, G, T = 0, 0, 0, 0
        if str(x) == "A":
            A = 1
        elif str(x) == "C":
            C = 1
        elif str(x) == "G":
            G = 1
        elif str(x) == "T":
            T = 1
        else:
            return x
        out.extend((A, C, G, T))
    return out


# Preprocessing
for window in range(flank_size, len_chr - flank_size, window_size):
    n_windows += 1   
    
    seq = reference.sequence(chrom, window - flank_size, window + flank_size + 1)
    if "N" in seq:
        n_windows_dropped_N += 1
        continue
    
    values = track.stats(chrom, window - flank_size, window + flank_size + 1, numpy = True)
    if np.isnan(np.min(values)):
        n_windows_dropped_nan += 1
        continue
        
    else:
        map_seq = mapping(seq, window - flank_size, window + flank_size + 1)
        outfile_seq.append(map_seq)
        outfile_values.append(track.stats(chrom, window - flank_size, window + flank_size + 1, exact=True)[0])


print("Windows size:", window_size)        
print("Total number of windows:", n_windows)
print("Total number of windows dropped with N in seq:", n_windows_dropped_N)
print("Total number of windows dropped with NaN in values:", n_windows_dropped_nan)


#Create dataframe, join files and save them in tsv
outfile = pd.DataFrame(outfile_seq)
outfile["values"] = outfile_values

outfile.to_csv("processed2_bed_info.bed", sep='\t', index=False, header = False)
                                    
                                    
                                    
                                    
                                    
                                    