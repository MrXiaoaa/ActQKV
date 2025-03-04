import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import glob
import os  # 确保在文件顶部导入os模块

FOLDER_PATH = "./result/needle/llama3-qllm-repr4-init64-l128-bs64-topk5-w1"
MODEL_NAME = "llama3"

def main():
    # Path to the directory containing JSON results
    folder_path = FOLDER_PATH
    if("/" in folder_path):
        model_name = folder_path.split("/")[-2]
    else: 
        model_name = MODEL_NAME
    print("model_name = %s" % model_name)

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}/*.json")
    print("json_files",json_files)
    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, 'r') as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            # score = json_data.get("score", None)
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()

            # print("length1",len(set(model_response.split()).intersection(set(expected_answer))))
            # print(set(model_response.split()).intersection(set(expected_answer)))
            # print("length2", len(expected_answer))
            # print(set(expected_answer))
            score = len(set(model_response.split()).intersection(set(expected_answer))) / len(set(expected_answer))
            #print([word for word in model_response.split() if word in expected_answer])
            # Appending to the list
            data.append({
                "Document Depth": document_depth,
                "Context Length": context_length,
                "Score": score
            })

    # Creating a DataFrame
    df = pd.DataFrame(data)
    
    locations = list(df["Context Length"].unique())
    locations.sort()

    for index, row in df.iterrows():
        print(row)
    print("Overall score %.3f" % df["Score"].mean())

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index() # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )


    # More aesthetics
    model_name_ = MODEL_NAME
    plt.title(f'Pressure Testing {model_name_} \nFact Retrieval Across Context Lengths ("Needle In A HayStack")')  # Adds a title
    plt.xlabel('Token Limit')  # X-axis label
    plt.ylabel('Depth Percent')  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    
    save_path = f"{FOLDER_PATH}/img/{model_name}.png"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 如果目录不存在则创建
    
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=150)

if __name__ == "__main__":
    main()
