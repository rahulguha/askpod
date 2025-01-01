import os
import glob

from util import *
from transformers import BartTokenizer, BartForConditionalGeneration

def chunk_text_with_sliding_window(text, tokenizer_name="facebook/bart-large-cnn", max_tokens=1024, overlap=100):
    """
    Splits text into chunks with a sliding window to ensure token continuity.

    Parameters:
        text (str): The input text to be chunked.
        tokenizer_name (str): The name of the tokenizer to use.
        max_tokens (int): Maximum number of tokens per chunk.
        overlap (int): Number of overlapping tokens between consecutive chunks.

    Returns:
        list of str: List of text chunks.
    """
    if overlap >= max_tokens:
        raise ValueError("Overlap must be smaller than max_tokens.")

    # Load the tokenizer
    tokenizer = BartTokenizer.from_pretrained(tokenizer_name)

    # Tokenize the input text
    tokens = tokenizer.encode(text, return_tensors=None)

    # Split tokens into chunks with overlap
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = tokens[start:end]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
        start += max_tokens - overlap

    return chunks

def summarize_chunks(chunks, model_name="facebook/bart-large-cnn"):
    """
    Summarizes each chunk of text using a pre-trained BART model.

    Parameters:
        chunks (list of str): List of text chunks to summarize.
        model_name (str): The name of the model to use for summarization.

    Returns:
        str: The concatenated summary of all chunks.
    """
    # Load the model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=130, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)


# def strip_extension(filename):
#     """
#     Strips the extension from a filename.
    
#     Args:
#     filename (str): The filename including the extension.
    
#     Returns:
#     str: The filename without the extension.
#     """
#     return os.path.splitext(filename)[0]
# def strip_before_last_slash(string):
#     return string.rsplit('/', 1)[-1]

# def create_file(filename, content):
    
#     with open(filename, "w") as f:   # Opens file and casts as f 
#         f.write(content )       # Writing

transcription_folder = "txt/"
summery_folder = "summary/"

folders = glob.glob(transcription_folder + "*.*") # read all files
print(len(folders))
documents = []

for folder in folders:
    summery_file_name = summery_folder + "summ_" + strip_extension(strip_before_last_slash(folder)) + ".txt"
    # print("***" + summery_file_name)
    if not os.path.exists(summery_file_name):
        with open(folder, 'r') as file:
            content = file.read()
            # print (len(content))
            # print (content)
            print (f"***** start summarization for {folder} - ({len(content)}) ***************")
            chunks = chunk_text_with_sliding_window(content)
            # print(len(chunks))
            summ = summarize_chunks(chunks)
            print (f"**** finish summarization for {folder} ***************")
            create_file(summery_file_name, summ)
            print(f"***** Summery created for {folder} to {summery_file_name}")
    
