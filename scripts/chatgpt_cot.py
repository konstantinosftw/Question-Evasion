# Import the openai package
import openai
import os
import pickle 
import argparse


import pandas as pd
def load_qevasion_dataset(add_specific_labels):
        
    dataset = pd.read_csv("../dataset/test_set.csv")
    texts = []
    for _, row in dataset.iterrows():
        
        if add_specific_labels:
            
            text = """Based on a segment of the interview in which the interviewer poses a series of questions, classify the type of response provided by the interviewee for the following question using the following taxonomy and then provide a chain of thought explanation for your decision:
1.1 Explicit - The information requested is explicitly stated (in the requested form)
1.2 Implicit - The information requested is given, but without being explicitly stated (not in the requested form)
2.1 Dodging - Ignoring the question altogether
2.2 Deflection - Starts on topic but shifts the focus and makes a different point than what is asked
2.3 Partial/half-answer - Offers only a specific component of the requested information.
2.4 General - The information provided is too general/lacks the requested specificity.
2.6 Declining to answer - Acknowledge the question but directly or indirectly refusing to answer at the moment
2.7 Claims ignorance - The answerer claims/admits not to know the answer themselves.
2.8 Clarification - Does not provide the requested information and asks for clarification.

You are required to respond with a single term corresponding to the taxonomy code as well as the chain of thought explanation.\nLet's think step by step\n ### Part of the interview ###\n""" + row["Interview Question"] + "\n" + row["Interview Answer"] + "\n\n" + "### Question ###\n\n"
        else:
            text = """Based on a segment of the interview in which the interviewer poses a series of questions, classify the type of response provided by the interviewee for the following question using the following taxonomy and then provide a chain of thought explanation for your decision:
1. Direct Reply - The information requested is explicitly stated (in the requested form)
2. Direct Non-Reply - The information requested is not given at all due to ignorance, need for clarification or declining to answer
3. Indirect Reply - The information requested is given in an incomplete way e.g. the answer is too general, partial, implicit, contradictory, diffused, dodging or deflection

You are required to respond with a single term corresponding to the taxonomy code as well as the chain of thought explanation.\nLet's think step by step\n ### Part of the interview ###\n""" + row["Interview Question"] + "\n" + row["Interview Answer"] + "\n\n" + "### Question ###\n\n"
            
        
        if add_specific_labels:
            text += row["Question"] + "\nTaxonomy code: " 
        else:
            text += row["Question"] + "\nTaxonomy code: "
        texts.append(text)

    print (texts[1])
    return texts


def extract_label(pred, add_specific_labels):
    if add_specific_labels:
        if "1.1" in pred:
            return '1.1 Explicit'
        elif "1.2" in pred:
            return '1.2 Implicit'
        elif "2.1" in pred:
            return '2.1 Dodging'
        elif "2.2" in pred:
            return '2.2 Deflection'
        elif "2.3" in pred:
            return '2.3 Partial/half-answer'
        elif "2.4" in pred:
            return '2.4 General'
        elif "2.5" in pred:
            return '2.5 Contradictory'
        elif "2.6" in pred:
            return '2.6 Declining to answer'
        elif "2.7" in pred:
            return '2.7 Claims ignorance'
        elif "2.8" in pred:
            return '2.8 Clarification'
        elif "2.9" in pred:
            return '2.9 Diffusion'
    else:
        if "1" in pred:
            return "Direct Reply"
        elif "2" in pred:
            return "Direct Non-Reply"
        elif "3" in pred:
            return "Indirect"
        
        if "direct reply" in pred.lower():
            return "Direct Reply"
        elif "direct non-reply" in pred.lower():
            return "Direct Non-Reply"
        elif "indirect" in pred.lower():
            return "Indirect" 



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Zero Shot gpt3.5 turbo")
    parser.add_argument("--api_key", type=str, help="OpenAI api key")
    parser.add_argument("--output_file", type=int, help="File that will contain the predicted labels from model")
    parser.add_argument("--add_specific_labels", action="store_true", help="Include this flag to indicate whether specific labels (e.g. General, Partial etc. - evasion problem) should be added or not. ")


    args = parser.parse_args()

    
    # Set openai.api_key to the OPENAI environment variable
    openai.api_key = args.api_key
    
    add_specific_labels = args.add_specific_labels
    texts = load_qevasion_dataset(add_specific_labels=add_specific_labels)
    
    mapping ={
        '1.1 Explicit': 'Direct Reply',
        '1.2 Implicit': 'Indirect',
        '2.1 Dodging': "Indirect",
        '2.2 Deflection': "Indirect",
        '2.3 Partial/half-answer': "Indirect",
        '2.4 General': "Indirect",
        '2.5 Contradictory': "Indirect",
        '2.6 Declining to answer': "Direct Non-Reply",
        '2.7 Claims ignorance': "Direct Non-Reply",
        '2.8 Clarification': "Direct Non-Reply",
        '2.9 Diffusion': "Indirect",
    }
    
    indexes, pred_labels = [], []
    for i, prompt in enumerate(texts):
    
        try:
            response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt},
                        ])
    
            resp = extract_label(response.choices[0].message.content, add_specific_labels)
            print (i, resp)
        
            indexes.append(i)
            pred_labels.append(resp)
        except Exception as e:
            print ("error", e)
    
    with open (args.output_file, "wb") as handle:
        pickle.dump([indexes, pred_labels], handle)
    
