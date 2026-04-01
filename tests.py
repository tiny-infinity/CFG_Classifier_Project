import pyBigWig 

bw = pyBigWig.open("projectData/hg38.phastCons7way.bw")
print(bw.chroms())