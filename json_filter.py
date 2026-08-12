import json
import string 

node = "Prediction.json"
edges = "predicted_edges.json"

with open(node, "r") as f:
    prediction_data = json.load(f)

print(len(prediction_data))

with open(edges, "r") as f:
    information = json.load(f)


def Label_search(filepath: str):
    for item in prediction_data:
        if item["filepath"] == filepath:
            return item["all_labels"][0]["label"]
    return None


result = []      # instead of data = []

for edge in information:
    source_id = edge["source_id"].split("::")[-1]
    target_id = edge["target_id"].split("::")[-1]

    label = Label_search(source_id)

    print(label)
    print("Source:", repr(source_id))
    break

print(string.punctuation)