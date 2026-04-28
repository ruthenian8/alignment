#!/usr/bin/env python
# coding: utf-8
"""
Read a plaintext audio index.
Cut the audio by timestamps from the index.
Convert the plaintext index to tsv.
"""

import subprocess
import re
import zipfile
import os
import sys
import shlex

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from bs4.element import Tag
# from google.colab import files


# In[2]:


def read_docx(filename:str) -> BeautifulSoup:
    with zipfile.ZipFile(filename, "r") as zipf:
        zipf.extractall(os.getcwd())
    with open(os.path.join(os.getcwd(), "word/document.xml"), "r", encoding="utf-8") as file:
        content:str = file.read()
    soup = BeautifulSoup(content, "html.parser")
    return soup


# In[5]:


def get_paragraph_strings(soup:BeautifulSoup) -> List[Tag]:
    paragraph:List[Tag] = [i for i in soup.find_all("w:p")]
    return [i.text for i in paragraph]


# In[137]:


def codes_from_paragraph(paragraph:List[str]) -> List[dict]:
    texts:List[str] = [i for i in paragraph if re.match(r"[\d:\.,]{12}[ -–]", i.strip())]
    conts:List[str] = []
    for text in texts:
        cleaned = clean = re.sub(r"[^\d\w]*$", "", text)
        lookup = re.search(r"[\d:\.,]{12}$", cleaned)
        conts.append(lookup.group().replace(",", ".") if lookup and re.search(r"продолж", text, re.IGNORECASE) else "")
    isTranscribed:List[str] = [False if "РАСПИСАНО" in i else True for i in texts]
    codes:List[str] = [re.match(r"^[\d:\.,]{12}", i.strip()).group().replace(",", ".") for i in texts]
    return [dict(start=codes[i], trans=isTranscribed[i], cont=conts[i], prev="", text=texts[i]) for i in range(len(codes))]

# In[142]:


def transform_df(codes:dict):
    df = pd.DataFrame.from_records(codes)
    for id_, row in df.iterrows():
        if row["cont"] != "":
            idx = np.where(df["start"].values == row["cont"])[0]
            df.loc[idx,"prev"] = int(id_)
    return df


# In[41]:


class Mapper():
    """Initialize with the name of the audio file"""
    def __init__(self, filename: str) -> None:
        print(filename)
        self.audio_path = Path(filename).resolve()
        self.filename = filename
        self.ext = self.audio_path.suffix
        self.base_name = self.audio_path.stem
        self.transcript_path = self.audio_path.parent.parent / "indices" / (self.base_name + ".txt")
        self.dir_path = self.audio_path.parent / self.base_name
        self.dir_path.mkdir(exist_ok=True)
        self.table = None
        self.parse_txt_file(str(self.transcript_path))
        # self.parse_docx_file(self.base_name)
        self.is_processed = False
    
    def parse_txt_file(self, file: str, save=False) -> None:
        with open(file) as fhandle:
            paragraph_strings = fhandle.read().splitlines()
        codes = codes_from_paragraph(paragraph_strings)
        for i in range(len(codes)):
            codes[i].update({"name": self.base_name + "No" + str(i) + self.ext})
        self.table = transform_df(codes)
        if save:
            self.save()

    def parse_docx_file(self, file: str, save=False) -> None:
        try:
            soup = read_docx(file + ".docx")
        except:
            raise OSError(f"file not found: {file + '.docx'}")
        paragraph_strings = get_paragraph_strings(soup)
        codes = codes_from_paragraph(paragraph_strings)
        for i in range(len(codes)):
            codes[i].update({"name": self.base_name + "No" + str(i) + self.ext})
        self.table = transform_df(codes)
        if save:
            self.save()
        
    def process_file(self, download=False) -> None:
        if self.table is None:
            return
        names = self.table["name"].tolist()
        codes = self.table["start"].tolist()
        if len(names) == 0: return
        for idx in range(len(names) - 1):
            output = shlex.quote(str(self.dir_path / (self.base_name + "No" + str(idx) + self.ext)))
            cmd = [
                "ffmpeg",
                "-y",
                "-i", shlex.quote(self.filename),
                "-ss", codes[idx],
                "-to", codes[idx+1],
                "-c", "copy",
                "-avoid_negative_ts", "1",
                output
            ]
            subprocess.run(cmd, check=True, capture_output=True)

        output = shlex.quote(str(self.dir_path / (self.base_name + "No" + str(len(codes) - 1) + self.ext)))
        cmd = [
                "ffmpeg",
                "-y",
                "-i", shlex.quote(self.filename),
                "-ss", codes[-1],
                "-c", "copy",
                "-avoid_negative_ts", "1",
                output
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        self.is_processed = True
            
    def reverse_concat(self, save=False):
        if not self.is_processed:
            return
        revers = self.table.sort_index(ascending=False)
        for _, row in revers.iterrows():
            if row["prev"] == "":
                continue
            prev_ = "\'" + str(self.dir_path / revers.loc[row["prev"], "name"]) + "\'"
            next_ = "\'" + str(self.dir_path / row["name"]) + "\'"
            sub = f'file {prev_}\\nfile {next_}'
            temporary = "/tmp/" + revers.loc[row["prev"], "name"]
            command = f'echo "{sub}" > $PWD/temp.txt; ffmpeg -y -f concat -safe 0 -i $PWD/temp.txt -c copy {temporary}; mv {temporary} {str(self.dir_path / revers.loc[row["prev"], "name"])};'
            os.system(command)
            print(command)
        else:
            print("done")
            self.do_cleanup()
        if save:
            self.save()
# subprocess.call('/bin/bash -c "$GREPDB"', shell=True, env={'GREPDB': 'echo 123'})            
    def do_cleanup(self):
        self.table = self.table.loc[self.table["prev"] == ""]
        for filename in os.listdir(str(self.dir_path)):
            if filename not in self.table["name"].values:
                command = f"rm {shlex.quote(str(self.dir_path / filename))}"
                os.system(command)

    def save(self):
        self.table.to_excel(str(self.dir_path / (self.base_name + ".xlsx")), index=True)


# In[ ]:


def main(directory):
    files = [i for i in os.listdir(directory)]
    for file2parse in files:
        if not re.search(r"\.wav$|\.WAV$", file2parse, re.IGNORECASE):
            continue
        try:
            mapper = Mapper(str(Path(directory) / file2parse))
            mapper.process_file()
            mapper.reverse_concat()
            mapper.save()
        except Exception as e:
            raise e
            # print(e)
            print(file2parse)
            sys.exit(1)
    else:
        sys.exit(0)


# In[ ]:


if __name__ == "__main__":
    main(sys.argv[1])


