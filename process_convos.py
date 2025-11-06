import json
from tqdm import tqdm

with open('conversations.json', 'r') as f:
    conversations = json.load(f)

def get_msg(convo):
    mapping = convo['mapping']
    user_msgs = []
    for node in mapping.keys():
        if mapping[node]['message']:
            if 'parts' in mapping[node]['message']['content'].keys() and mapping[node]['message']['author']['role'] == 'user':
                if isinstance(mapping[node]['message']['content']['parts'][0], str):
                    user_msgs.append(mapping[node]['message']['content']['parts'])
    return user_msgs