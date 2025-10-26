import pandas as pd
import pybedtools

# promotors = pd.read_csv("chr22_promoters.bed", sep='\t', header=None)
# promotors = promotors.drop_duplicates(subset=1)
# promotors.to_csv("promotors_unique.bed", sep='\t', index=False, header = False)

promotors = pybedtools.BedTool("promotors_unique.bed")
predictions = pybedtools.BedTool("../../group_MM/Models/NN/predictions_header.bed")


#promotors_pred = predictions.intersect(promotors, u=True, header=True, f=1).saveas("promoters_preds.bed")
non_promotors_pred = predictions.intersect(promotors, v=True, header=True).saveas("non_promoters_preds.bed")
