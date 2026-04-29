import sys
from pathlib import Path
import pandas as pd


def read_into_table(filename):
    with open(filename) as file:
        file_content = file.read().rstrip("\n")
    record_strings = file_content.split("\n\n")
    record_lines = [item.splitlines() for item in record_strings]
    n_interviewers = [len(item[2].split(",")) for item in record_lines]
    n_interviewees = [len(item[-1].split(",")) for item in record_lines]
    n_speakers = [i + j for i, j in zip(n_interviewers, n_interviewees)]
    select_r_lines = [item[3:-1] for item in record_lines]
    new_line_strings = [" ".join(item) for item in select_r_lines]
    return {"id": list(range(1, len(new_line_strings)+1)), "transcript": new_line_strings, "max_speakers": n_speakers, "min_speakers": n_interviewees}


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        sys.exit(1)
    filename = args[-1]
    new_line_strings = read_into_table(filename)
    new_line_dataframe = pd.DataFrame(new_line_strings)
    new_path = Path(filename).with_suffix(".tsv")
    new_line_dataframe.to_csv(str(new_path), sep="\t", index=False)
    
