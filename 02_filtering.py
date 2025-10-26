import pybedtools

file = pybedtools.BedTool("../../group_MM/Preprocessing/processed2_bed_401.bed")
# hg38 = BedTool("~/nnPib2021/data/masks/hg38.umap.tar.gz")
k100_merged = pybedtools.BedTool("../../data/masks/hg38/k100.umap.merged.bed")

filtered = file.intersect(k100_merged, u=True, header = True, f=1).saveas("filtered_401.bed")  
