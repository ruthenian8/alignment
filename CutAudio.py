#!/usr/bin/env python
# coding: utf-8

# In[56]:

# import subprocess
import shlex
import numpy as np
import pandas as pd
import re
import zipfile
import os
import sys
from bs4 import BeautifulSoup
from typing import List
from bs4.element import Tag
#from google.colab import files


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
        self.audio_path = Path(filename).resolve()
        self.filename = filename
        self.ext = self.audio_path.suffix
        self.base_name = self.audio_path.stem
        self.transcript_path = self.audio_path.parent.parent / "indices" / (self.base_name + ".txt")
        self.table = None
        self.parse_txt_file(str(self.transcript_path))
        # self.parse_docx_file(self.base_name)
        self.is_processed = False
    
    def parse_txt_file(self, file: str, save=False) -> None:
        with open(file) as fhandle:
            paragraph_strings = fhandle.read().splitlines()
        codes = codes_from_paragraph(paragraph_strings)
        for i in range(len(codes)):
            codes[i].update({"name":file + "No" + str(i) + self.ext})
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
            codes[i].update({"name":file + "No" + str(i) + self.ext})
        self.table = transform_df(codes)
        if save:
            self.save()
        
    def process_file(self, download=False) -> None:
        if self.table is None:
            return
        names = self.table["name"].tolist()
        codes = self.table["start"].tolist()
        if len(names) == 0: return
        if not os.path.isdir(self.base_name):
            os.system(f"mkdir {shlex.quote(self.base_name)}")
        for idx in range(len(names) - 1):
            output = shlex.quote(os.path.join(self.base_name, self.base_name+ "No" +str(idx)+self.ext))
            command = f"ffmpeg -ss {codes[idx]} -i {shlex.quote(self.filename)} -to {codes[idx+1]} -c copy -avoid_negative_ts 1 {output}"
            os.system(command)
            # print(command)
        output = shlex.quote(os.path.join(self.base_name, self.base_name+str(len(codes)-1)+self.ext))
        command = f"ffmpeg -ss {codes[-1]} -i {shlex.quote(self.filename)} -c copy -avoid_negative_ts 1 {output}"
        # print(command)
        os.system(command)
        self.is_processed = True
        if download:
            command = f"zip -r {shlex.quote('/content/'+self.base_name+'.zip')} . -i {shlex.quote('./'+self.base_name+'*')}"
            os.system(command)
            files.download(self.base_name + ".zip")
            
    def reverse_concat(self, save=False):
        if not self.is_processed:
            return
        revers = self.table.sort_index(ascending=False)
        for _, row in revers.iterrows():
            if row["prev"] == "":
                continue
            prev_ = "\'$PWD/" + os.path.join(self.base_name, revers.loc[row["prev"], "name"]) + "\'"
            next_ = "\'$PWD/" + os.path.join(self.base_name, row["name"]) + "\'"
            sub = f'file {prev_}\\nfile {next_}'
            temporary = "/tmp/" + revers.loc[row["prev"], "name"]
            command = f'echo "{sub}" > $PWD/temp.txt; ffmpeg -y -f concat -safe 0 -i $PWD/temp.txt -c copy {temporary}; mv {temporary} {"$PWD/" + os.path.join(self.base_name, revers.loc[row["prev"], "name"])};'
            print(command)
            os.system(command)
        else:
            print("done")
            self.do_cleanup()
        if save:
            self.save()
# subprocess.call('/bin/bash -c "$GREPDB"', shell=True, env={'GREPDB': 'echo 123'})            
    def do_cleanup(self):
        self.table = self.table.loc[self.table["prev"] == ""]
        for filename in os.listdir(self.base_name):
            if filename not in self.table["name"].values:
                command = f"rm {shlex.quote(os.path.join(self.base_name, filename))}"
                os.system(command)

    def save(self):
        self.table.to_excel(os.path.join(self.base_name, self.base_name + ".xlsx"), index=True)


# In[ ]:


def main(directory):
    files = os.listdir(directory)
    for file2parse in files:
        if not re.search(r"\.wav$|\.WAV$", file2parse, re.IGNORECASE):
            continue
        try:
            mapper = Mapper(file2parse)
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


