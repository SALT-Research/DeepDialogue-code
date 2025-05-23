from transformers import AutoFeatureExtractor
from torch.utils.data import Dataset
import torchaudio
import random
import torch


class EmotionDataset(Dataset):
    def __init__(
        self, 
        data, 
        sample_rate=16000, 
        max_audio_duration=10.0,
        feature_extractor=None,
        ):
        self.data = data
        self.sample_rate = sample_rate
        self.max_audio_duration = max_audio_duration
        self.feature_extractor = feature_extractor
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        ##################################################################
        ##########               Store the label                ##########
        ##################################################################
        label = self.data.iloc[idx]['label']
        ##################################################################


        ##################################################################
        #######                  Load the audio                    #######
        ##################################################################
        audio_path = self.data.iloc[idx]['segment_id']
        try:
            waveform, sr = torchaudio.load(
                f'DeepDialogue-xtts/{audio_path}'
                )
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            ## Read random audio file
            random_audio_path = self.data.iloc[random.randint(0, len(self.data)-1)]['segment_id']
            waveform, sr = torchaudio.load(
                f'DeepDialogue-xtts/{random_audio_path}'
                )
            print(f"Loaded random audio file {random_audio_path} instead.")
        ##################################################################


        ##################################################################
        #######    Resample to the target sample rate if needed    #######
        ##################################################################
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        ##################################################################


        ##################################################################
        #######                  Extract features                  #######
        ##################################################################
        inputs = self.feature_extractor(
            waveform.squeeze(),
            sampling_rate=self.feature_extractor.sampling_rate, 
            return_tensors="pt",
            max_length=int(self.feature_extractor.sampling_rate * self.max_audio_duration), 
            truncation=True,
            padding='max_length'
            )
        inputs = inputs['input_values'].squeeze()
        ##################################################################

        return inputs, label