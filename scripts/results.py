import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

mapping ={
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


filenames = ["../results/llama-2-7b_qaevasion_clarity_test_set.pickle",
             "../results/llama-2-13b_qaevasion_clarity_test_set.pickle",
             "../results/llama_2_70b_clarity_zero_shot_test_set.pickle",
             "../results/falcon_40b_clarity_zero_shot_test_set.pickle",
             "../results/gpt3.5_evasion_zero_shot_test_set.pickle",
             "../results/gpt3.5_clarity_zero_shot_cot_test_set.pickle",

             
             "../results/llama-2-7b_qaevasion_evasion_test_set.pickle",
             "../results/llama-2-13b_qaevasion_evasion_test_set.pickle",
             "../results/llama_2_70b_evasion_zero_shot_test_set.pickle",
             "../results/llama-2-70b_qaevasion_evasion_test_set.pickle",
             "../results/falcon_7b_qaevasion_evasion_test_set.pickle",
             "../results/falcon_40b_evasion_zero_shot_test_set.pickle",
             "../results/falcon_40b_qaevasion_evasion_test_set.pickle",
             "../results/gpt3.5_evasion_zero_shot_test_set.pickle", 
              "../results/gpt3.5_evasion_zero_shot_cot_test_set.pickle",
             
             ]
             
test_set = pd.read_csv("../dataset/Inter-Annotator/test_set.csv")

true_labels = []

for _, row in test_set.iterrows():
    labels = [mapping[row["Annotator1"]], mapping[row["Annotator2"]], mapping[row["Annotator3"]]]
    true_labels.append(max(set(labels), key=labels.count))

for filename in filenames:
    
    with open(filename, "rb") as handle:
        index, pred_labels = pickle.load(handle)
        
    
    tt, pp = [], []
    for i in range(len(true_labels)):
        t = true_labels[i]
        
        if i in index:
            p = str(pred_labels[index.index(i)])
        
        if p in mapping:
            p = mapping[p]
            
        if "zero" not in filename:
            if p in list(mapping.values()):
                tt.append(t)
                pp.append(p)
        else:
            tt.append(t)
            pp.append(p)

    print (filename)
    print (round (accuracy_score(tt, pp), 4), end = "\t")
    print (round(precision_score(tt, pp, average='macro'), 4), end = "\t")
    print (round(recall_score(tt, pp, average='macro'), 4), end = "\t")
    print (round(f1_score(tt, pp, average='macro'), 4))
    print ("---")