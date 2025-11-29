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


def get_paras(soup:BeautifulSoup) -> List[Tag]:
    paras:List[Tag] = [i for i in soup.find_all("w:p")]
    return paras


# In[137]:


def codes_from_paras(paras:List[Tag]) -> List[dict]:
    filtered:List[Tag] = [i for i in paras if re.match(r"[\d:\.,]{12}[ -–]", i.text.strip())]
    # text_entries:List[str] = []
    # for para in filtered:
        # pieces = [x.text for x in para.find_all("w:t")]
        # good_pieces = [n for n in filter(lambda x: False if re.search(r"начало|продолжение|расписано", x, re.IGNORECASE) or re.search(r"[\d:\.,]{12}", x) else True, pieces)]
        # text_entries.append(" ".join(good_pieces).replace("  ", " "))
    texts:List[str] = [i.text for i in filtered]
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


def code_to_seconds(code:str):
    integral_part = code[:8].split(":")
    hours = int(integral_part[0])
    minutes = int(integral_part[1])
    seconds = int(integral_part[2])
    milli = int(re.search(r"\d{2,3}$", code).group())
    if milli > 500:
        seconds += 1
    seconds += minutes * 60
    seconds += hours * 3600
    return seconds


# In[158]:


class Mapper():
    """Initialize with the name of the audio file"""
    def __init__(self, filename:str) -> None:
        self.filename = filename
        self.ext = re.search(r"\..+?$", filename).group()
        self.shortened = re.sub("\..+?$", "", filename)
        self.table = None
        self.parse_file(self.shortened)
        self.is_processed = False
    
    def parse_file(self, file:str, save=False) -> None:
        try:
            soup = read_docx(file + ".docx")
        except:
            raise OSError(f"file not found: {file + '.docx'}")
        paras = get_paras(soup)
        codes = codes_from_paras(paras)
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
        if not os.path.isdir(self.shortened):
            os.system(f"mkdir {shlex.quote(self.shortened)}")
        for idx in range(len(names) - 1):
            range_ = str(code_to_seconds(codes[idx+1]) - code_to_seconds(codes[idx]))
            output = shlex.quote(os.path.join(self.shortened, self.shortened+ "No" +str(idx)+self.ext))
            command = f"ffmpeg -ss {codes[idx]} -i {shlex.quote(self.filename)} -t {range_} -c copy -avoid_negative_ts 1 {output}"
            os.system(command)
            # print(command)
        output = shlex.quote(os.path.join(self.shortened, self.shortened+str(len(codes)-1)+self.ext))
        command = f"ffmpeg -ss {codes[-1]} -i {shlex.quote(self.filename)} -c copy -avoid_negative_ts 1 {output}"
        # print(command)
        os.system(command)
        self.is_processed = True
        if download:
            command = f"zip -r {shlex.quote('/content/'+self.shortened+'.zip')} . -i {shlex.quote('./'+self.shortened+'*')}"
            os.system(command)
            files.download(self.shortened + ".zip")
            
    def reverse_concat(self, save=False):
        if not self.is_processed:
            return
        revers = self.table.sort_index(ascending=False)
        for _, row in revers.iterrows():
            if row["prev"] == "":
                continue
            prev_ = "\'$PWD/" + os.path.join(self.shortened, revers.loc[row["prev"], "name"]) + "\'"
            next_ = "\'$PWD/" + os.path.join(self.shortened, row["name"]) + "\'"
            sub = f'file {prev_}\\nfile {next_}'
            temporary = "/tmp/" + revers.loc[row["prev"], "name"]
            command = f'echo "{sub}" > $PWD/temp.txt; ffmpeg -y -f concat -safe 0 -i $PWD/temp.txt -c copy {temporary}; mv {temporary} {"$PWD/" + os.path.join(self.shortened, revers.loc[row["prev"], "name"])};'
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
        for filename in os.listdir(self.shortened):
            if filename not in self.table["name"].values:
                command = f"rm {shlex.quote(os.path.join(self.shortened, filename))}"
                os.system(command)

    def save(self):
        self.table.to_excel(os.path.join(self.shortened, self.shortened + ".xlsx"), index=True)


# In[ ]:


def main(directory):
    files = os.listdir(directory)
    for file2parse in files:
        if not re.search(r"\.wma$|\.mp3$", file2parse, re.IGNORECASE):
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


# In[138]:


# ag = read_docx("BIF_.docx")
# paras = get_paras(ag)
# codes = codes_from_paras(paras)
# bif = transform_df(codes)
# bif.sort_index(ascending=False)
# bif.loc[bif.loc[3,"prev"],:]
# np.where(bif["start"].apply(lambda x: x not in bif["cont"].values) == False)
# a="KIL&KVI_1.WMA"
# f"\'$PWD/{a}\'"
# bif.loc[bif["prev"].isna()]

