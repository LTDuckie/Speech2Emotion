#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.nn as nn
import librosa  # audio file related library
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf
import random


# In[2]:


# ----------------- dataset class -----------------
class EmotionDataset(Dataset):
    def __init__(self, file_list, labels, sr=22050, n_mels=64):
        # initialize files, labels, sample rate, Mel Spectrograms(visualization of audio frequency), duration, label encoder, encoded labels
        self.file_list = file_list # file_list
        self.labels = labels # emotion labels(in string)
        self.sr = sr # sample rate
        self.n_mels = n_mels # Mel_num in the Spectrogram of the audio
        self.encoder = LabelEncoder() # the encoder for turning string labels into num labels
        self.labels_encoded = self.encoder.fit_transform(labels) # encoded labels in nums
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file = self.file_list[idx]
        y, sr = librosa.load(file, sr=self.sr)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels, n_fft=2048, hop_length=512)
        mel_db = librosa.power_to_db(mel)
        t_mel_db = torch.tensor(mel_db) # Convert the numpy array into tensor
        N_mel_db = (t_mel_db-torch.mean(t_mel_db))/ torch.std(t_mel_db) # data normalization
        padded_N_mel_db= F.pad(N_mel_db, (0,240-N_mel_db.shape[1],0,0)) # only padded time axis to make their shape the same as (64, 240).
        mel_db_resized = padded_N_mel_db[np.newaxis, ...] # add an axis for CNN input 
        return torch.tensor(mel_db_resized, dtype=torch.float32), torch.tensor(self.labels_encoded[idx], dtype=torch.long)


# In[3]:


import glob
files = glob.glob("ravdess/**/*.wav", recursive=True)
labels = [f.split("-")[2] for f in files]  # extract labels
dataset = EmotionDataset(files, labels)
dataset[82]


# In[4]:


# #----------------- CNN model----------------
class CNNEmotion(nn.Module):
    def __init__(self, num_classes):
        super(CNNEmotion, self).__init__()
        self.net = nn.Sequential( # 1*num*t, set as 1*64*240
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 16*32*120
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 32*16*60
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # 64*8*30
            nn.Flatten(), # 1600
            nn.Linear(64 * 8 * 30, 64), nn.ReLU(), # 64*240
            nn.Linear(64, num_classes) # 8*1920
        )
    def forward(self, x):
        return self.net(x)


# In[5]:


# ----------------- get files and labels from RAVDESS -----------------
def get_ravdess_data(data_dir):
    emotions_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    files, labels = [], []
    for root, _, filenames in os.walk(data_dir): # traverse files
        for file in filenames:
            if file.endswith('.wav'):
                emotion_code = file.split('-')[2] # the third num in audio files in RAVDESS stands for emotion labels
                if emotion_code in emotions_map:
                    files.append(os.path.join(root, file))
                    labels.append(emotions_map[emotion_code])
    return files, labels


# In[ ]:




