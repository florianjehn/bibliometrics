import pandas as pd
import json
import os

with open(f'data{os.sep}raw{os.sep}vos_cluster_and_graph.json', 'rb') as f:  # Open in binary mode
    data_bytes = f.read()

# Decode the bytes using utf-8-sig to handle potential BOM
data = json.loads(data_bytes.decode('utf-8-sig'))

# Extract items from the network key
items = data['network']['items']

# Extract the links
links = data['network']['links']

# Define an empty list to store dictionaries for the dataframe
meta_df_data = []

# Loop through each item in the 'items' list
for item in items:
    # Extract relevant information from each item
    vos_id = item['id']
    authors = [i.title() for i in item['description'].split(
        'Title:'
    )[0].replace(
        "<td>", ""
    ).replace(
        "</td>", ""
    ).replace(
        "<tr>", ""
    ).replace(
        "</tr>", ""
    ).replace(
        "<table>", ""
    ).replace(
        "</table>", ""
    ).replace(
        "Authors:", ""
    ).strip().split(";")]
    title = item['description'].split(
        'Title:'
    )[1].split(
        '</tr>'
    )[0].strip(
    ).replace("</td>", "").replace("<td>", "").strip().title()
    year = int(item['scores']['Pub. year'])
    citations = int(item['scores']['Citations'])
    source = item['description'].split(
        'Source:'
    )[1].split('year')[0].split(",")[0].split(">")[-1].strip().capitalize()
    cluster = item['cluster']

    weights = dict(item['weights'])
    for key, value in weights.items():
        weights[key] = float(value)

    # Create a dictionary for each item
    item_dict = {
        'VOS_ID': vos_id,
        'Authors': authors,
        'Title': title,
        'Year': year,
        'Citations': citations,
        'Source': source,
        'Cluster': cluster

    }

    # Add the weights to the dictionary
    item_dict.update(weights)

    # Append the dictionary to the list
    meta_df_data.append(item_dict)

# Create the pandas dataframe from the list of dictionaries
meta_df = pd.DataFrame(meta_df_data)

# Create a second df for the links
links_df_data = []

# Loop through each link in the 'links' list
for link in links:
    # Extract relevant information from each link
    source = int(link['source_id'])
    target = int(link['target_id'])
    weight = float(link['strength'])

    # Create a dictionary for each link
    link_dict = {
        'VOS_Source_ID': source,
        'VOS_Target_ID': target,
        'Strength': weight
    }

    # Append the dictionary to the list
    links_df_data.append(link_dict)

# Create the pandas dataframe from the list of dictionaries
links_df = pd.DataFrame(links_df_data)

# Save the dataframes to csv files
meta_df.to_csv(f'data{os.sep}prepared{os.sep}vos_meta.csv', index=False)
links_df.to_csv(f'data{os.sep}prepared{os.sep}vos_links.csv', index=False)