#!/usr/bin/env python
# coding: utf-8

# In[85]:


import glob
import librosa


# In[87]:


if __name__ == "__main__":
    files = glob.glob("ravdess/**/*.wav", recursive=True)
    labels = [f.split("-")[2] for f in files]  # extract labels


# In[89]:


len(files), files[1], labels[1]


# In[91]:


import IPython.display as ipd
test_audio_path = files[1]
ipd.Audio(test_audio_path)


# In[93]:


import librosa
y, sr = librosa.load(test_audio_path)


# In[95]:


y.shape, sr


# In[97]:


import librosa.display
import matplotlib.pyplot as plt   

plt.figure(figsize=[15,2])   
librosa.display.waveshow(y=y)


# In[119]:


plt.figure(figsize=[15,30])

# Mel Spectrogram without Log transformation
# t_window = n_fft/sr, hop_length = x_window2 - x_window1, t_hop = hop_length/sr, n_mels: features for extraction
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048,hop_length=512,n_mels=64)
plt.subplot(1,2,1)
plt.title("Mel Spectrogram")
plt.imshow(mel_spectrogram)


# Log-Mel Spectrogram
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
plt.subplot(1,2,2)
plt.title("Log-Mel Spectrogram")
plt.imshow(log_mel_spectrogram)


# In[127]:


log_mel_spectrogram.shape


# In[129]:


import torch

t_log_mel_spectrogram = torch.tensor(log_mel_spectrogram) # Convert the numpy array into tensor
N_log_mel_spectrogram = (t_log_mel_spectrogram-torch.mean(t_log_mel_spectrogram))/ torch.std(t_log_mel_spectrogram)
print(f"The Log-Mel Spectrogram before normalization:\n\n {t_log_mel_spectrogram[30:60, 50:90]}.\n\n")
print(f"The Log-Mel Spectrogram after normalization:\n\n {N_log_mel_spectrogram[30:60, 50:90]}.")


# In[38]:


import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf
import random


# In[41]:


random_list=[] # Generate a random list.
for i in range(4):
    random_list.append(random.randint(0,1440))

img_num=1
plt.figure(figsize=[15,15])
for i in random_list:
    audio, sr = librosa.load(files[i]) # Randomly select 4 audio data in RAVDESS database.
    
    # Present the original Log-Mel spectrogram.
    original_mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=64)
    original_log_mel_spectrogram = librosa.power_to_db(original_mel_spectrogram)
    plt.subplot(4,2,img_num)
    plt.axis("off")
    plt.imshow(original_log_mel_spectrogram)
    plt.title("Original Spectrogram")
    
    # Present the augmented Log-Mel spectrogram.
    flow = naf.Sequential([
        nas.FrequencyMaskingAug(zone=(0, 1), factor=(4, 8), verbose=4),
        nas.TimeMaskingAug(zone=(0.5, 0.15))])
    aug_data = flow.augment(original_log_mel_spectrogram)
    plt.subplot(4,2,img_num+1)
    plt.axis("off")
    plt.imshow(aug_data[0])
    plt.title("Augmented Spectrogram")
    
    img_num+=2


# In[43]:


import torch.nn.functional as F


# In[45]:


max_duration = 0


# In[55]:


for i in files:
    y1, sr1 = librosa.load(i)
    duration = len(y1)/sr1
    if duration > max_duration:
        max_duration = duration


# In[135]:


hop_length = 512
max_duration / (hop_length / sr)


# In[59]:


plt.figure(figsize=[15,8])

# Padding step
padding_log_mel_spectrogram= F.pad(N_log_mel_spectrogram, (0,240-N_log_mel_spectrogram.shape[1],0,0))

plt.subplot(1,2,1)
plt.title("Original Log-Mel Spectrogram")
plt.imshow(N_log_mel_spectrogram)

plt.subplot(1,2,2)
plt.title("Padding Log-Mel Spectrogram")
plt.imshow(padding_log_mel_spectrogram)


# In[ ]:




