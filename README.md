# Speech2Emotion
This project is set up to make a "conversation to motions" system with the help of [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture) which requires emotion input as a parameter to generate motions from audios.
## Getting Started
Set up the [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture) project as they instructed.
Then

    cd ./DiffuseStyleGesture\main\mydiffusion_zeggs

Put all files of this project into ```./DiffuseStyleGesture\main\mydiffusion_zeggs``` and run

    pip install -r requirements.txt


## Prepare Dataset
We use the RAVDESS dataset, which contains 24 professional actors (12 male, 12 female) vocalizing two lexically-matched statements in a neutral North American accent. Speech recordings are acted in eight emotional states: neutral, calm, happy, sad, angry, fearful, disgust, and surprised. Audio files are recorded in 16-bit, 48kHz format, and in this work are downsampled to 22.05kHz for feature extraction.
You can run

    python ravdess_download.py

Or, you can directly download this dataset from [RAVDESS](https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip)

## Train the model

    
    
