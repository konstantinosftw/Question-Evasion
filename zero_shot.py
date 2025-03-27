from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle
import argparse
import pandas as pd


mapping = {
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


def load_qevasion_dataset(add_specific_labels):
    
    
    dataset = pd.read_csv("dataset/Inter-Annotator/test_set.csv")
    texts = []
    for _, row in dataset.iterrows():
        
        if add_specific_labels:
            
            text = """Based on a segment of the interview in which the interviewer poses a series of questions, classify the type of response provided by the interviewee for the following question using the following taxonomy:
1.1 Explicit - The information requested is explicitly stated (in the requested form)
1.2 Implicit - The information requested is given, but without being explicitly stated (not in the requested form)
2.1 Dodging - Ignoring the question altogether
2.2 Deflection - Starts on topic but shifts the focus and makes a different point than what is asked
2.3 Partial/half-answer - Offers only a specific component of the requested information.
2.4 General - The information provided is too general/lacks the requested specificity.
2.5 Contradictory - The response makes conflicting statements.
2.6 Declining to answer - Acknowledge the question but directly or indirectly refusing to answer at the moment
2.7 Claims ignorance - The answerer claims/admits not to know the answer themselves.
2.8 Clarification - Does not provide the requested information and asks for clarification.
2.9 Diffusion - Points out that the question is based on false hypotheses and does not provide the requested information.

You are required to respond with a single term corresponding to the taxonomy code.\n\n ### Part of the interview ###\n""" + row["Interview Question"] + "\n" + row["Interview Answer"] + "\n\n" + "### Question ###\n\n"
        else:
            text = """Based on a segment of the interview in which the interviewer poses a series of questions, classify the type of response provided by the interviewee for the following question using the following taxonomy:
1. Direct Reply - The information requested is explicitly stated (in the requested form)
2. Direct Non-Reply - The information requested is not given at all due to ignorance, need for clarification or declining to answer
3. Indirect Reply - The information requested is given in an incomplete way e.g. the answer is too general, partial, implicit, contradictory, diffused, dodging or deflection

You are required to respond with a taxonomy code and only that.\n\n ### Part of the interview ###\n""" + row["Interview Question"] + "\n" + row["Interview Answer"] + "\n\n" + "### Question ###\n\n"
            
        
        if add_specific_labels:
            text += row["Question"] + "\nTaxonomy code: " # + row["label"] + "\n\n"
        else:
            text += row["Question"] + "\nTaxonomy code: " # + mapping[row["label"]] + "\n\n"
        texts.append(text)

    print (texts[1])
    return texts



def generate(input_text, model, tokenizer, max_new_tokens = 15):
    input_ids = tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=True).to("cuda")
    output = model.generate(input_ids, max_new_tokens=max_new_tokens, num_return_sequences=1) 
    return tokenizer.decode(output[0], skip_special_tokens=True)


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
        
        
def main(model_name, output_file, add_specific_labels):

    
    output_file = f"results/{output_file}_zero_shot_test_set.pickle"

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 device_map='auto', 
                                                 load_in_4bit=True, 
                                                 )
    # load dataset 
    texts = load_qevasion_dataset(add_specific_labels)

    indexes, pred_labels = [], []
    for index in range (0, len(texts)):

        try:

            text = texts[index]
            pred = generate(text, model, tokenizer, 3)
            pred_label = pred.split("\nTaxonomy code: ")[1]
            pred_label = extract_label(pred_label, add_specific_labels)

            if add_specific_labels:
                pred_label = mapping[pred_label]


            pred_labels.append(pred_label)
            indexes.append(index)
            print (index, pred_label)

        except Exception as e:
            pred_labels.append(None)
            indexes.append(index)
            print (f"Exception: {e}")

    with open (output_file, "wb") as handle:
        pickle.dump([indexes, pred_labels], handle)
        
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Inference script with arguments.")
    parser.add_argument("--model_name", type=str, help="Name of the model e.g. bigscience/bloom-3b")
    parser.add_argument("--output_file", type=str, help="The name of the fine tuned model")
    parser.add_argument("--add_specific_labels", action="store_true", help="Include this flag to indicate whether specific labels (e.g. General, Partial etc) should be added or not")
    
    args = parser.parse_args() 
    main(args.model_name, args.output_file, args.add_specific_labels)