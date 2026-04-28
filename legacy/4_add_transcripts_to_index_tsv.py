"""
Read transcripts from tsv, read the index from tsv. Add the transcripts to the index as a column.
"""
import sys

import pandas as pd


def main(infile1, infile2, outfile):
    file_index = pd.read_csv(infile1, sep=",", index_col=0)
    transcripts = pd.read_csv(infile2, sep="\t", index_col=0)
    file_index = file_index.sort_values(["trans", "start"], ascending=[False, True])
    transcripts.index = file_index.index[:transcripts.shape[0]]
    result = file_index.join(transcripts)
    result.to_csv(outfile, sep=",", index=None)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        exit(1)
    infile1 = sys.argv[1]
    infile2 = sys.argv[2]
    outfile = infile1.rstrip(".csv") + ".1" + ".csv"
    main(infile1, infile2, outfile)
