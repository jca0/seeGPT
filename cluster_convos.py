from sklearn.cluster import KMeans, SpectralClustering, HDBSCAN
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
import json
from pathlib import Path

load_dotenv()

output_folder = Path("user_data")
output_folder.mkdir(parents=True, exist_ok=True)

with open(output_folder / 'all_user_msgs.json', 'r') as f:
    all_user_msgs = json.load(f)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embeddings = np.load(output_folder / 'embeddings.npy')

print("Clustering")
# clustering = KMeans(n_clusters=5, random_state=42)
# clustering = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=42)
clustering = HDBSCAN(min_cluster_size=30, metric='cosine', cluster_selection_method='eom')
clustering.fit(embeddings)
labels = clustering.labels_
print("Done clustering")

def show_cluster_info(labels, output_file='label_to_msg_mapping.json'):
    """creates mapping of cluster labels to messages"""
    label_to_msg_mapping = {}
    label_count = {}
    for label in np.unique(labels):
        label_count[label] = len(np.where(labels == label)[0])
        indices = np.where(labels == label)[0]
        messages = [all_user_msgs[i] for i in indices]
        label_to_msg_mapping[int(label)] = messages
    with open(output_folder / output_file, 'w') as f:
        json.dump(label_to_msg_mapping, f)
    print(f"Saved label_to_msg_mapping to {output_folder / output_file}")
    return label_to_msg_mapping, label_count

def name_clusters(label_to_msg_mapping):
    cluster_names = {i: None for i in label_to_msg_mapping.keys()}
    for i in tqdm(sorted(label_to_msg_mapping.keys())):
        cluster_msgs = label_to_msg_mapping[i][:500]
        cluster_msgs_text = " | ".join([msg for msg in cluster_msgs])
        prompt = f"""You are analyzing a cluster of similar user messages. Based on the examples below, create a concise descriptive label (3-5 words) that captures the main topic or theme.

            Messages in this cluster:
            {cluster_msgs_text}

            Instructions:
            - Identify the common theme or topic
            - Use 2-4 words
            - Be specific and descriptive
            - Return only the label name

            Cluster name:
        """

        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
        )
        output = response.choices[0].message.content
        cluster_names[i] = output
    with open(output_folder / 'cluster_names.json', 'w') as f:
        json.dump(cluster_names, f)
    print(f"Saved cluster_names to {output_folder / 'cluster_names.json'}")
    return cluster_names

if __name__ == "__main__":
    label_to_msg_mapping, label_count = show_cluster_info(labels)
    print(f"Cluster sizes: \n {label_count}")
    cluster_names = name_clusters(label_to_msg_mapping)
    print(f"Cluster names: \n {cluster_names}")
