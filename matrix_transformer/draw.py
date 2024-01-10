import os
import re
import pandas as pd
import matplotlib.pyplot as plt

def read_and_parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    sections = content.split('Running perf stat for')
    data = {}

    for section in sections[1:]:
        lines = section.split('\n')
        test_type = lines[0].strip()
        for line in lines[1:]:
            match = re.search(r'(\d+(?:,\d+)*)\s+(.+?)\s+', line)
            if match:
                value, metric = match.groups()
                value = int(value.replace(',', ''))
                data[(test_type, metric)] = value
    return data

# Assuming all result files are in a folder named 'Transformer_Accelerator/matrix_transformer'
folder_path = '/users/LiuQun/Transformer_Accelerator/matrix_transformer'
result_files = [f for f in os.listdir(folder_path) if f.startswith('result_') and f.endswith('.txt')]

all_data = {}

for file in result_files:
    file_path = os.path.join(folder_path, file)
    file_data = read_and_parse_file(file_path)
    index = int(re.search(r'result_(\d+).txt', file).group(1))
    all_data[index] = file_data

# Creating separate DataFrames
original_df = pd.DataFrame()
block_df = pd.DataFrame()

for index, data in all_data.items():
    for (test_type, metric), value in data.items():
        if 'original' in test_type:
            original_df.at[index, metric] = value
        elif 'Block' in test_type:
            block_df.at[index, metric] = value

# Simple Plot (You can modify this part to plot specific metrics)
def plot_data(df, title):
    plt.figure(figsize=(10, 6))
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

plot_data(original_df, 'Original Data')
plot_data(block_df, 'Block Data')

