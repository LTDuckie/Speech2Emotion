# Speech2Emotion to Support Conversation2Motion
This project is set up to make a "conversation to motions" system with the help of [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture) which requires emotion input as a parameter to generate motions from audios.  
The video of running the code is available at [GoogleDrive](https://drive.google.com/file/d/1Ll5Wp5w-fRmoJERyManGLCRcR5e9g2qI/view?usp=sharing).  
The video of demo result is available at [Bilibili](https://www.bilibili.com/video/BV1mDtSzzEEi/).  

## Getting Started
Set up the [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture) project as they instructed.
Then

    cd ./DiffuseStyleGesture/main/mydiffusion_zeggs

Put all files of this project into ```./DiffuseStyleGesture/main/mydiffusion_zeggs``` and exchange the ```sample.py``` file with the new one here. You can still directly use ```sample.py``` after this.
Run

    pip install -r requirements.txt


## Prepare Dataset
We use the RAVDESS dataset, which contains 24 professional actors (12 male, 12 female) vocalizing two lexically-matched statements in a neutral North American accent. Speech recordings are acted in eight emotional states: neutral, calm, happy, sad, angry, fearful, disgust, and surprised. Audio files are recorded in 16-bit, 48kHz format, and in this work are downsampled to 22.05kHz for feature extraction.  
You can run

    python ravdess_download.py

Or, you can directly download this dataset from [RAVDESS](https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip) and extract them into this folder.  
If the dataset is well prepared, you should be able to use ```data_preprocess_sample.py``` and visualize the data of audios in the dataset if you run the code step by step.  

## Train and test the model
In ```ER_model.py```, we define the funtion to extract the information in the audio and the structure of the model, you can change the parameters in this file if you need.  
For training process, do

    python train_and_test.py

You should get a ```best_emotion_model.pth``` file and a ```classes.pt``` file after you run the code above.  

## Try to use
First, create an empty folder named ```speech2emotion``` and extract ```example_audio.zip``` inside.  
I - mydiffusion_zeggs  
I - train_and_test.py  
I - preprocess_and_generate.py  
I - ...  
I - speech2emotion  
  II -- example_audio   
    III --- Angry_neutral  
    III --- Happy_neutral  
    III -- ...  

### Test the emotion recognition part
To only test the emotion recognition part, put some wav files into ```example_audio``` and run:  

    python ER_sample.py

The result will show the emotion recognition results of these wav files.
You may also change the detection path in line 49 of ```ER_sample.py```.

### Combine with motion generation.
There are three conversation sample folders in ```example_audio```. We can recognize their emotions and generate motions by running

    python preprocess_and_generate.py ./speech2emotion/example_audio/Angry_neutral

You may change the audio path if you put them at other places.

    python preprocess_and_generate.py path_to_your_audio_folder

And the results would be saved to ```./mydiffusion_zeggs/sample_dir/generating_time/```.  

Below are the results of generated motions from ```./speech2emotion/example_audio/Angry_neutral```.  

<img width="2109" height="830" alt="image" src="https://github.com/user-attachments/assets/4ff4eb57-b61b-472e-9844-4b179a688ea0" />

With text transcript:  
”He took my laptop without asking again.”  
”Did you talk to him about it?”  
”Yes. But he just laughed. I’m so mad.”  
”Maybe you should lock your things.”  

## References
[DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture)


    
    
