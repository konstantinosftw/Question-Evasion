import numpy as np
import pandas as pd
import math
import os

from statsmodels.stats.inter_rater import fleiss_kappa
from statsmodels.stats.inter_rater import aggregate_raters
from sklearn.metrics import cohen_kappa_score
from itertools import combinations

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def calculate_fleiss_kappa(labels1, labels2, labels3):
    """
    Calculates Fleiss Kappa score for 3 lists of annotations (labels).
    """
    # Combine the labels into a list of tuples
    labels = list(zip(labels1, labels2, labels3))

    # Calculate the Fleiss Kappa score
    kappa = fleiss_kappa(aggregate_raters(labels, n_cat=None)[0])

    return kappa
    

taxonomy ={
    '1.1 Explicit': 'Direct Reply',
    '1.2 Implicit': 'Indirect',
    '2.1 Dodging': "Indirect",
    '2.2 Deflection': "Indirect",
    '2.3 Partial/half-answer': "Indirect",
    '2.4 General': "Indirect",
    '2.6 Declining to answer': "Direct Non-Reply",
    '2.7 Claims ignorance': "Direct Non-Reply",
    '2.8 Clarification': "Direct Non-Reply",
}



dataset = pd.read_csv("../dataset/QAEvasion.csv")

anns = {}
for i in range (len(dataset)):
    annt = dataset.iloc[i]["annotator_id"]
    if annt not in anns:
        anns[annt] = 0
    anns[annt] += 1


number_unique_conf = len(set(dataset["url"]))
print (f"Number of Unique Conferences in the dataset: {number_unique_conf}")
print (f"Number of Annotations: {len(dataset)}")
print (f"Number of Annotators: {len(anns)}")
mean_num_annotations = math.ceil (np.mean(list(anns.values())))
print (f"Mean number of Annotations per Annotator: {mean_num_annotations}")


# Inter-Annotator Agreement 
# Load data
annotations = {}
for filename in ["Annotator1.csv", "Annotator2.csv", "Annotator3.csv"]:
    filename = os.path.join("dataset/Inter-Annotator", filename)
    df = pd.read_csv(filename)
    annotator = filename.split("/")[-1].split(".csv")[0]
    annotations[annotator] = {(row["Interview Question"], row["Interview Answer"], row["Question"]): row["Label"] for _, row in df.iterrows()}


# Fleiss Kappa
print ("\n ---- Results for 13 Labels ---- ")
preds = {k: [] for k in annotations}
annotators = list(annotations.keys())
all_keys = set([i for k in annotators for i in annotations[k]])
for k in all_keys:
    found_all = True
    for v in annotations.values():
        if k not in v:
            found_all = False
    if found_all:
        for ann in annotations:
            preds[ann].append(annotations[ann][k])

class_labels = list(taxonomy.keys())
annotator1 = preds["Annotator1"]
annotator2 = preds["Annotator2"]
annotator3 = preds["Annotator3"]

values = ['1.1 Explicit',
 '1.2 Implicit',
 '2.1 Dodging',
 '2.2 Deflection',
 '2.3 Partial/half-answer',
 '2.4 General',
 '2.6 Declining to answer',
 '2.7 Claims ignorance',
 '2.8 Clarification',]

cmm = []
for c1 in values:
    row = []
    for c2 in values:
                

        if c1 == c2:
            row.append(1)
            continue

        classes = [c1, c2]

        # Select instances where both annotators labeled as 1 or 2
        label_1_indices = [i for i in range(len(annotator1)) if annotator1[i] in classes and annotator2[i] in classes and annotator3[i] in classes]

        if len(label_1_indices) == 0:
            row.append(np.nan)
            continue


        annotator1_selected = [annotator1[i] for i in label_1_indices]
        annotator2_selected = [annotator2[i] for i in label_1_indices]
        annotator3_selected = [annotator3[i] for i in label_1_indices]
        ll = []
        for a, b, c in zip(annotator1_selected, annotator2_selected, annotator3_selected):
            ll += [a,b,c]

        if len(set(classes) - set(ll)) != 0:
            row.append(np.nan)
            continue
           

        fk = calculate_fleiss_kappa(annotator1_selected, annotator2_selected, annotator3_selected)
        # if np.isnan(fk):
        #     row.append(round(5, 4))
        # else:
        row.append(round(fk, 4))
    cmm.append(row.copy())




print ("Plot confusion matrix for Evasion Problem")
plt.figure(figsize=(10, 8))
ticks = [" ".join(v.split(" ")[1:]) for v in values]
sns.heatmap(cmm, annot=True, fmt=".2f", cmap="Blues", vmin=0, xticklabels=ticks, yticklabels=ticks)
plt.title('Evasion Confusion Fleiss Matrix')
plt.savefig('evasion_confusion_fleiss_matrix.png')
plt.show()


data = np.array([v for v in preds.values()]).T
kappa = fleiss_kappa(aggregate_raters(data, n_cat=None)[0])
print(f"Fleiss' kappa for 13 Labels: {kappa:.3f}")


# print ("Cohen Kappa for between each Annotator")
# combinations_result = list(combinations(annotators, 2))
# cohen = {}
# for k1, k2 in combinations_result:
#     kappa = cohen_kappa_score(preds[k1], preds[k2])
#     cohen[(k1, k2)] = kappa
#     cohen[(k2, k1)] = kappa

# print ("         & Annotator1 & Annotator2 & Annotator3  \\\\ \hline")
# for k1 in ["Annotator1", "Annotator2", "Annotator3"]:
#     print (k1, end = "   &   ")
#     for k2 in ["Annotator1", "Annotator2", "Annotator3"]:
#         if k1 == k2:
#             print (1, end = "      &    ")
#         elif k2 != "Annotator3": 
#             print (round (cohen[(k1, k2)], 4), end = "       &       ")
#         else:
#             print (round (cohen[(k1, k2)], 4), end = "  ")
#     print ("\\\\   \hline")


print ("\n ---- Analysis for Completeness Problem (Direct Reply, Direct Non-Reply and Indirect) ---- ")

ppp = {}
for ann in preds:
    ppp[ann] = []
    for v in preds[ann]:
        if v in taxonomy:
            ppp[ann].append(taxonomy[v])

preds = ppp
data = np.array([v for v in preds.values()]).T
kappa = fleiss_kappa(aggregate_raters(data, n_cat=None)[0])
print(f"Fleiss' kappa for ambiguity problem (3 Labels): {kappa:.3f}")

print ("Cohen Kappa for between each Annotator")
combinations_result = list(combinations(annotators, 2))
cohen = {}
for k1, k2 in combinations_result:
    kappa = cohen_kappa_score(preds[k1], preds[k2])
    cohen[(k1, k2)] = kappa
    cohen[(k2, k1)] = kappa

print ("         & Annotator1 & Annotator2 & Annotator3  \\\\ \hline")
for k1 in ["Annotator1", "Annotator2", "Annotator3"]:
    print (k1, end = "   &   ")
    for k2 in ["Annotator1", "Annotator2", "Annotator3"]:
        if k1 == k2:
            print (1, end = "      &    ")
        elif k2 != "Annotator3": 
            print (round (cohen[(k1, k2)], 4), end = "       &       ")
        else:
            print (round (cohen[(k1, k2)], 4), end = "  ")
    print ("\\\\   \hline")

print ()
print (" --- Fleiss Score for each one of the 3 labels --- ")
annotator1 = preds["Annotator1"]
annotator2 = preds["Annotator2"]
annotator3 = preds["Annotator3"]

values = ["Direct Reply", "Direct Non-Reply", "Indirect"]
cmm = []
for c1 in values:
    row = []
    for c2 in values:


        if c1 == c2:
            row.append(1)
            continue

        classes = [c1, c2]

        # Select instances where both annotators labeled as 1 or 2
        label_1_indices = [i for i in range(len(annotator1)) if annotator1[i] in classes and annotator2[i] in classes and annotator3[i] in classes]

        if len(label_1_indices) == 0:
            row.append(0)
            continue

        annotator1_selected = [annotator1[i] for i in label_1_indices]
        annotator2_selected = [annotator2[i] for i in label_1_indices]
        annotator3_selected = [annotator3[i] for i in label_1_indices]


        fk = calculate_fleiss_kappa(annotator1_selected, annotator2_selected, annotator3_selected)
        row.append(round(fk, 4))
    cmm.append(row.copy())

print ("         & Direct Reply & Direct Non-Reply & Indirect  \\\\ \hline")
for i, k1 in enumerate(["Direct Reply", "Direct Non-Reply", "Indirect"]):
    print (k1, end = "   &   ")
    for j, k2 in enumerate(["Direct Reply", "Direct Non-Reply", "Indirect"]):
        if k1 == k2:
            print (1, end = "      &    ")
        elif k2 != "Annotator3": 
            print (round (cmm[i][j], 4), end = "       &       ")
        else:
            print (round (cmm[i][j], 4), end = "  ")
    print ("\\\\   \hline")

