import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import moviepy.editor as mp





def extract_audio_from_video(path,output_dir):
    
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    # Create the output directory if it doesn't exist
    
    os.makedirs(output_dir, exist_ok=True)

    for f in onlyfiles:
        file_name, _ = os.path.splitext(f)
        input_file_path = os.path.join(path, f)
        output_file_path = os.path.join(output_dir, '{}.wav'.format(file_name))

        clip = mp.VideoFileClip(input_file_path)
        clip.audio.write_audiofile(output_file_path)
        
    return




    
if __name__ == '__main__':
    path = 'data/video/context_final'
    output_dir = './data/audio/context_final_audio/'
    extract_audio_from_video(path,output_dir)
