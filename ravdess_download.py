#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import zipfile
import requests


# In[9]:


# ----------------- 1. download and unzip RAVDESS -----------------
def download_ravdess(url, save_path="ravdess.zip", extract_dir="ravdess"):
    if not os.path.exists(extract_dir): # check if ravdess folder already exists
        print("Downloading RAVDESS dataset...")
        r = requests.get(url, stream=True) # use streaming download to save memory
        with open(save_path, 'wb') as f: # write file
            for chunk in r.iter_content(chunk_size=1024): # 1KB per iteration
                if chunk:
                    f.write(chunk)
        print("Extracting dataset...") # extracting
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Done!")
    else:
        print("RAVDESS dataset already exists.")
    return extract_dir


# In[11]:


if __name__ == "__main__":
    RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip" # URL of RAVDESS dataset
    DATA_DIR = download_ravdess(RAVDESS_URL)


# In[ ]:





# In[ ]:




