import json 
import pickle
import re
from thefuzz import process
from thefuzz import fuzz
# from word2number import w2n

taxonomy = [
    "1.1 Explicit",
    "1.2 Implicit",
    "2.1 Dodging",
    "2.2 Deflection",
    "2.3 Partial/half-answer",
    "2.4 General",
    "2.6 Declining to answer",
    "2.7 Claims ignorance",
    "2.8 Clarification",
]

taxonomy_with_descr = [
    "1.1 Explicit - The information requested is explicitly stated (in the requested form)",
    "1.2 Implicit - The information requested is given, but without being explicitly stated (not in the requested form)",
    "2.1 Dodging - Ignoring the question altogether",
    "2.2 Deflection - Starts on topic but shifts the focus and makes a different point than what is asked",
    "2.3 Partial/half-answer - Offers only a specific component of the requested information.",
    "2.4 General - The information provided is too general/lacks the requested specificity.",
    "2.5 Contradictory - The response makes conflicting statements.",
    "2.6 Declining to answer - Acknowledge the question but directly or indirectly refusing to answer at the moment",
    "2.7 Claims ignorance - The answerer claims/admits not to know the answer themselves.",
    "2.8 Clarification - Does not reply and asks for clarification on the question.",
    "2.9 Diffusion - The answerer points out that the information requested does not exist (the answer renders the question invalid)",

]
    
def extract_labels(text):
    # Define a regular expression pattern to match lines starting with 'Label: something'
    pattern = r'(Label|Verdict|Reply):\s(.+)'

    # Use re.findall to find all matches in the text
    matches = re.findall(pattern, text, re.MULTILINE)

    labels = []
    # Print the list of "somethings"
    for match in matches:
        prefix, label = match
        labels.append(label)

    return labels


def connect_to_taxonomy(labels):
    tax_labels = []
    for r in labels:
        t, _ = process.extract(r, taxonomy_with_descr, scorer=fuzz.ratio)[0]
        tax_labels.append(t)
    return tax_labels


def extract_taxonomy_labels(response):
    labels = extract_labels(response)
    return connect_to_taxonomy(labels)


def find_similar_text_position(big_text, small_text, threshold=80):
    big_text_words = big_text.split()
    small_text_words = small_text.split()

    for i in range(len(big_text_words) - len(small_text_words) + 1):
        window = " ".join(big_text_words[i:i + len(small_text_words)])
        similarity = fuzz.ratio(" ".join(small_text_words), window)
        if similarity >= threshold:
            return i  # Return the position if a match is found
    return -1  # Return -1 if no match is found


class GPT:
    
    def __init__(self, filename):
        with open(filename, "rb") as handle:
            self.annotations = pickle.load(handle)
            
        self.dataset = {}
        for ann in self.annotations:
            self.dataset[(ann[1], ann[2])] = ann
            
    
    def get_responses(self):
        presidents = {}

        for row, q, a, [anl, resp] in self.annotations:

            president = row["president"]
            if president not in presidents:
                presidents[president] = {}

            # labels = extract_taxonomy_labels(resp)

            labels = connect_to_taxonomy(extract_labels(resp))
            for l in labels:
                l = l.split(" -")[0].split(" (")[0]

                if l not in presidents[president]:
                    presidents[president][l] = 1
                else:
                    presidents[president][l] += 1
        return presidents


def find_position(main_text, sub_text):
    # Find the best match
    ratio = fuzz.ratio(main_text.lower(), sub_text.lower())

    # If the ratio is above a certain threshold, consider it a match
    if ratio >= 10:
        position = main_text.lower().find(sub_text.lower())
        return position
    else:
        return -1  # No match found

from difflib import SequenceMatcher
def find_start_position(main_text, sub_text, threshold=0.5):
    # Initialize SequenceMatcher with the two texts
    matcher = SequenceMatcher(None, main_text, sub_text)
    # Get the matching blocks
    matching_blocks = matcher.get_matching_blocks()
    # Find the block with the maximum size (similarity)
    max_block = max(matching_blocks, key=lambda x: x.size)
    # Calculate the similarity ratio
    similarity_ratio = max_block.size / len(sub_text)
    # Check if the similarity is above the threshold
    if similarity_ratio >= threshold:
        start_position = max_block.a
        return start_position
    else:
        return -1  # No match found
    

from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz

def find_similar_questions(big_text, target_question):
    # Load a pre-trained language model for sentence embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Define the target question
    target_question_embedding = model.encode(target_question)

    # Split the big text into segments based on 'Question part:'
    segments = re.split('Question part: ', big_text)[1:]

    # Find the segment with the highest similarity to the target question
    max_similarity = 0
    most_similar_segment = ""

    for segment in segments:
        # Extract the question part
        question_part = re.search(r'(.+?)\n', segment).group(1)

        # Encode the question part with the language model
        question_embedding = model.encode(question_part)

        # Calculate the cosine similarity between the target question and the current question part
        similarity = fuzz.ratio(target_question, question_part)

        # Update the most similar segment if the current one has higher similarity
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_segment = question_part

    return most_similar_segment.strip()
