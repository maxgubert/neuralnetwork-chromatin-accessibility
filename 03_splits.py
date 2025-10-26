import pandas as pd
from sklearn.model_selection import train_test_split

file = pd.read_csv("~/nnPib2021/group_MM/Filtering/filtered_401.bed", sep='\t', header=None)
file_train, file_valtest = train_test_split(file, test_size = 0.1)
file_val, file_test = train_test_split(file, test_size = 0.5)

file_train.to_csv("file_train_401.tsv", sep="\t")
file_val.to_csv("file_val_401.tsv", sep="\t")
file_test.to_csv("file_test_401.tsv", sep="\t")