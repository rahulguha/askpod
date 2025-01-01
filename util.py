import torch, whisper
import json
import os
import yt_dlp

def strip_extension(filename):
    """
    Strips the extension from a filename.
    
    Args:
    filename (str): The filename including the extension.
    
    Returns:
    str: The filename without the extension.
    """
    return os.path.splitext(filename)[0]
def check_file(file_name):
    if not os.path.exists(file_name):
        return False
    else: 
        return True
def strip_before_last_slash(string):
    return string.rsplit('/', 1)[-1]
def create_file(filename, content):
    
    with open(filename, "w") as f:   # Opens file and casts as f 
        f.write(content )       # Writing
def my_hook(d):
    if d['status'] == 'downloading':
        print ("###### downloading "+ str(round(float(d['downloaded_bytes'])/float(d['total_bytes'])*100,1))+"%")
    if d['status'] == 'finished':
        filename=d['###### filename']
        print(filename)

def transcribe (mp3_filename, transcription_filename):
    # assume mp3 exists
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model("tiny.en").to(device)
    txtfilepath = "txt"
    print (f"******* start transcription of {mp3_filename}")
    result = model.transcribe(mp3_filename)
    # writing transcription to file
    print (f"********* writing transcription to {transcription_filename}")
    create_file(transcription_filename, result["text"])
