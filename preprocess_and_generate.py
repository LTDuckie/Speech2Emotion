import os
import sys
import shutil
import glob
from datetime import datetime
import torch
import argparse
import librosa
import yaml
from easydict import EasyDict
from pprint import pprint

from ER_sample import predict_emotion
from ER_model import CNNEmotion
from sample import main_wrapper

# 配置路径
CLASSES_PATH = './classes.pt'
EMOTION_MODEL_PATH = './emotion_model.pth'
STYLE_MAP = {
    '03': 'Happy', # happy
    '04': 'Sad', # sad
    '01': 'Neutral', # neutral
    '05': 'Angry', # angry
    '02': 'Relaxed', # calm
    '06': 'Sad', # fearful
    '07': 'Angry', # disgust
    '08': 'Happy' # surprised
}


def init_emotion_model():
    classes = torch.load(CLASSES_PATH)
    model = CNNEmotion(num_classes=len(classes))
    model.load_state_dict(torch.load(EMOTION_MODEL_PATH))
    print(model, classes)
    return model, classes


def convert_and_copy_files(input_folder, target_base='./input_audio'):
    emotion_model, classes = init_emotion_model()

    # 创建目标子目录，例如 ./input_audio/example_audio/
    folder_name = os.path.basename(os.path.normpath(input_folder))
    target_folder = os.path.join(target_base, folder_name)
    os.makedirs(target_folder, exist_ok=True)

    wav_files = glob.glob(os.path.join(input_folder, "*.wav"))
    if not wav_files:
        print(f"No .wav files found in {input_folder}")
        sys.exit(1)

    print(f"Found {len(wav_files)} audio files. Predicting emotions and copying...")

    for idx, wav_path in enumerate(wav_files):
        emotion = predict_emotion(wav_path, emotion_model, classes).lower()
        style = STYLE_MAP.get(emotion, 'Neutral')
        new_filename = f"{str(idx+1).zfill(3)}_{style}_0_x_0_0.wav"
        target_path = os.path.join(target_folder, new_filename)
        shutil.copy(wav_path, target_path)
        print(f"[{os.path.basename(wav_path)}] -> {emotion} -> {style} -> {new_filename}")

    return target_folder


def run_sample_on_folder(wav_folder, config_path='./configs/DiffuseStyleGesture.yml', gpu='0', model_path='./model000450000.pt'):
    save_dir = 'sample_dir/'+str(datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=config_path)
    parser.add_argument('--gpu', type=str, default=gpu)
    parser.add_argument('--no_cuda', type=list, default=[gpu])
    parser.add_argument('--model_path', type=str, default=model_path)
    parser.add_argument('--audiowavlm_path', type=str, default='')
    parser.add_argument('--max_len', type=int, default=0)
    args = parser.parse_args([])

    # 加载配置
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    for k, v in vars(args).items():
        config_dict[k] = v
    config = EasyDict(config_dict)

    # 遍历该文件夹下所有 .wav 并运行 main_wrapper
    wav_files = glob.glob(os.path.join(wav_folder, '*.wav'))
    for wav_file in wav_files:
        print(f"\n=== Generating BVH for: {os.path.basename(wav_file)} ===")
        main_wrapper(config, wav_file, save_dir=save_dir)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python preprocess_and_generate.py ./speech2emotion/example_audio")
        sys.exit(1)

    input_path = sys.argv[1]
    processed_folder = convert_and_copy_files(input_path)
    run_sample_on_folder(processed_folder)
    # python preprocess_and_generate.py ./speech2emotion/example_audio

