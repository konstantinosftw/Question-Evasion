import json 
from utils import find_start_position, extract_labels, connect_to_taxonomy
from sklearn.metrics import accuracy_score


def extract_counterfactual_label(q, text):
    try:
        text = text.lower()
        q = q.lower()
        pos = find_start_position(text, q)
        if pos == -1:
            return None
        clabel = text[pos:].split("counterfactual label: ")[1].split("\n")[0]
        return clabel
    except:
        return None

def extract_original_label(q, text):
    try:
        text = text.lower()
        q = q.lower()
        pos = find_start_position(text, q)
        if pos == -1:
            return None
        clabel = text[pos:].split("original label: ")[1].split("\n")[0]
        return clabel
    except:
        return None




def find_disagreement(data):
    annots, clabels, tlabels = [], [], []
    for row in data:
        if "Reply" in row:
            for r in row["Reply"]:
                label = r["labels"][0].lower()
                tlabel = extract_original_label(r["text"], row["countersummary"])
                clabel = extract_counterfactual_label(r["text"], row["countersummary"])
                if clabel:
                    annots.append (label)
                    clabels.append(clabel)
                    tlabels.append(tlabel)
        if "Non-Reply" in row:
            for r in row["Non-Reply"]:
                label = r["labels"][0].lower()
                tlabel = extract_original_label(r["text"], row["countersummary"])
                clabel = extract_counterfactual_label(r["text"], row["countersummary"])
                if clabel:
                    annots.append (label)
                    clabels.append(clabel) 
                    tlabels.append(tlabel)

    return accuracy_score(annots, clabels)


filenames = [
    ["dataset/Counterfactual-Summaries/Annotator1_countersummaries.json", "Annotator1"],
    ["dataset/Counterfactual-Summaries/Annotator2_countersummaries.json", "Annotator2"],
    ["dataset/Counterfactual-Summaries/Annotator3_countersummaries.json", "Annotator3"],

]

for filename, ann in filenames:
    with open (filename, "r") as handle:
        data = json.load(handle)

        dis = find_disagreement(data)
        print (f"Dissagremment between {ann} and fake label of the summary produces by chatgpt is: {1 - round (dis, 4)}")
        