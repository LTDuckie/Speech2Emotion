#!/usr/bin/env python
# coding: utf-8

# In[9]:


import import_ipynb


# In[25]:


import torch
import librosa
import numpy as np
from ER_model import CNNEmotion
import torch.nn.functional as F
import glob


# In[17]:


def predict_emotion(audio_path, model, classes, sr=22050, n_mels=64):
    y, sr = librosa.load(audio_path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel)
    t_mel_db = torch.tensor(mel_db)
    N_mel_db = (t_mel_db-torch.mean(t_mel_db))/ torch.std(t_mel_db)
    padded_N_mel_db= F.pad(N_mel_db, (0,240-N_mel_db.shape[1],0,0))
    mel_db = torch.tensor(padded_N_mel_db[np.newaxis, np.newaxis, ...], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(mel_db)
        pred = torch.argmax(output, dim=1).item()

    return classes[pred]


# In[35]:


if __name__ == "__main__":
    classes = torch.load("classes.pt")
    model = CNNEmotion(num_classes=len(classes))
    model.load_state_dict(torch.load("emotion_model.pth"))
    
    files = glob.glob("example_audio/*.wav", recursive=True)
    print(files)
    for file in files:
        emotion = predict_emotion(file, model, classes)
        print(f"{file} -> {emotion}")


# In[23]:


classes


# In[ ]:


emotions_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }


# In[ ]:




