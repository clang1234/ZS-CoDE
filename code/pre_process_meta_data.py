import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import json

def create_meta_data(utterance_json,video_context_dir,video_utterance_dir,audio_context_dir,audio_utterance_dir):
    with open(utterance_json) as f:
        # Load the JSON data
        data = json.load(f)
        for key, value in data.items():
            value['video_context_path'] = os.path.join(video_context_dir, '{}_c.mp4'.format(key))
            value['video_utterance_path']= os.path.join(video_utterance_dir, '{}.mp4'.format(key))
            value['audio_context_path']= os.path.join(audio_context_dir, '{}_c.wav'.format(key))
            value['audio_utterance_path'] = os.path.join(audio_utterance_dir, '{}.wav'.format(key))
    
    file_path = "meta_data.json"

    # Write the dictionary to a JSON file
    with open(file_path, "w") as json_file:
        json.dump(data, json_file,indent=2)






utterance_json = r'data\text\Persona_added_utterances.json'
video_context_dir = r'data\video\context_final'
video_utterance_dir = r'data\video\utterances_final'
audio_utterance_dir = r'data\audio\utterances_final_audio'
audio_context_dir = r'data\audio\context_final_audio'
create_meta_data(utterance_json,video_context_dir,video_utterance_dir,audio_context_dir,audio_utterance_dir)
