# extract fbanck from wav and save to file
# pre processd an audio in 0.09912s

import os
import librosa
import numpy as np
import pandas as pd
from glob import glob
from python_speech_features import fbank, delta

import sid_constants as c
import silence_detector
from time import time
import sys
# print(sys.path)
# from constants import c.SAMPLE_RATE


np.set_printoptions(threshold=10)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)


def find_files(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True) 
    
def VAD(audio):
    chunk_size = int(c.SAMPLE_RATE*0.05) # 50ms
    index = 0
    sil_detector = silence_detector.SilenceDetector(15)
    nonsil_audio=[]
    while index + chunk_size < len(audio):
        if not sil_detector.is_silence(audio[index: index+chunk_size]):
            nonsil_audio.extend(audio[index: index + chunk_size])
        index += chunk_size

    return np.array(nonsil_audio)

def read_audio(filename, sample_rate=c.SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    print(audio.shape[0])
    audio = VAD(audio.flatten())
    print(audio.shape[0])
    start_sec, end_sec = c.TRUNCATE_SOUND_SECONDS
    start_frame = int(start_sec * sample_rate)
    end_frame = int(end_sec * sample_rate)

    if len(audio) < (end_frame - start_frame):
        au = [0] * (end_frame - start_frame)
        for i in range(len(audio)):
            au[i] = audio[i]
        audio = np.array(au)
    print(audio.shape[0])
    return audio

def normalize_frames(m,epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v),epsilon) for v in m]

def extract_features(signal=np.random.uniform(size=48000), target_sample_rate=c.SAMPLE_RATE):
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=64, winlen=0.025)  #filter_bank (num_frames , 64),energies (num_frames ,)
    filter_banks = normalize_frames(filter_banks)
    frames_features = filter_banks     # (num_frames , 64)
    num_frames = len(frames_features)  # (num_frames)
    return np.reshape(np.array(frames_features),(num_frames, 64, 1))   #(num_frames,64, 1)

def data_catalog(dataset_dir=c.DATASET_DIR, pattern='*.npy'): 
    libri = pd.DataFrame()                                            
    libri['filename'] = find_files(dataset_dir, pattern=pattern)
    libri['filename'] = libri['filename'].apply(lambda x: x.replace('\\', '/'))  # normalize windows paths
    libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-1].split('-')[0]) # x.split('/')[-1]->1-100-0001.wav 
    num_speakers = len(libri['speaker_id'].unique())
    print('Found {} files with {} different speakers.'.format(str(len(libri)).zfill(7), str(num_speakers).zfill(5)))
    return libri
    #                          filename                                       speaker_id
    #   0    audio/LibriSpeech100/train-clean-100/1/100/1-100-0001.wav        1
    #   1    audio/LibriSpeech100/train-clean-100/1/100/1-100-0002.wav        1

def prep(libri,out_dir=c.DATASET_DIR):
    # os.mkdir(out_dir)
    # i=0
    for i in range(len(libri)):
        filename = libri[i:i+1]['filename'].values[0] # for example : audio/LibriSpeech100/train-clean-100/1/100/1-100-0001.wav
        target_filename = os.path.join(out_dir, os.path.basename(filename).split('.')[0] + '.npy')
        print('preprocessing %s' % target_filename)
        # target_filename = out_dir + filename.split("/")[-1].split('.')[0] + '.npy' # for example :audio/LibriSpeech100/train-clean-100-npy/1-100-0001.npy
        fp = open(target_filename,'w')  
        fp.close()
        raw_audio = read_audio(filename)
        feature = extract_features(raw_audio, target_sample_rate=c.SAMPLE_RATE)
        if feature.ndim != 3 or feature.shape[0] < c.NUM_FRAMES or feature.shape[1] !=64 or feature.shape[2] != 1:
            print('there is an error in file:',filename)
            continue
        np.save(target_filename, feature)

def preprocess_and_save(wav_dir=c.WAV_DIR,out_dir=c.DATASET_DIR):

    orig_time = time()
    libri = data_catalog(wav_dir, pattern='**/*.wav') 

    print("Extract fbank from audio and save as npy")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    prep(libri,out_dir)
    print("Extract audio features and save it as npy file, cost {0} seconds".format(time()-orig_time))


if __name__ == '__main__':
    preprocess_and_save(c.WAV_DIR,c.DATASET_DIR)
    