import json
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import os
import numpy as np
from pathlib import Path

load_dotenv()

MAX_CONTEXT_LENGTH = 8192
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 50 # set to 35 for worst case
output_folder = Path("user_data")
output_folder.mkdir(parents=True, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open('conversations.json', 'r') as f:
    conversations = json.load(f)

def get_all_user_msgs(convo, output_file='all_user_msgs.json'):
    """
    gets all user messages from a conversation.json file
    """
    def get_msg(convo):
        mapping = convo['mapping']
        user_msgs = []
        for node in mapping.keys():
            if mapping[node]['message']:
                if 'parts' in mapping[node]['message']['content'].keys() and mapping[node]['message']['author']['role'] == 'user':
                    if isinstance(mapping[node]['message']['content']['parts'][0], str):
                        user_msgs.append(mapping[node]['message']['content']['parts'][0])
        return user_msgs
    
    all_user_msgs = []
    for convo in tqdm(conversations, desc="Processing conversations"):
        all_user_msgs.extend(get_msg(convo))
    
    # save all_user_msgs to a json file
    with open(output_folder / output_file, 'w') as f:
        json.dump(all_user_msgs, f)
    print(f"Saved all_user_msgs to {output_folder / output_file}")

    return all_user_msgs

def clean_msgs(msgs, max_length=MAX_CONTEXT_LENGTH, embedding_model=EMBEDDING_MODEL):
    """clean and truncate messages to be passed into embedding model"""
    cleaned_msgs = []
    for msg in tqdm(msgs, desc="Cleaning messages"):
        if len(msg) == 0: # empty message
            continue
        encoding = tiktoken.encoding_for_model(embedding_model)
        tokens = encoding.encode(msg, disallowed_special=set())
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]
        cleaned_msgs.append(encoding.decode(tokens))
    return cleaned_msgs
    
def embed_msgs(msgs, batch_size=BATCH_SIZE, embedding_model=EMBEDDING_MODEL, output_file='embeddings.npy'):
    """embed messages using the embedding model and save to a npy file"""
    embeddings = []
    for i in tqdm(range(0, len(msgs), batch_size), desc="Embedding messages"):
        batch = msgs[i:i+batch_size]
        try:
            response = client.embeddings.create(input=batch, model=embedding_model)
            embeddings.extend([data.embedding for data in response.data])
        except Exception as e:
            print(f"Error embedding batch {i}: {e}")
            return
    
    np.save(output_folder / output_file, embeddings)
    print(f"Saved embeddings to {output_folder / output_file}")
    return embeddings


if __name__ == "__main__":
    all_user_msgs = get_all_user_msgs(conversations)
    cleaned_msgs = clean_msgs(all_user_msgs)
    embedded_msgs = embed_msgs(cleaned_msgs)