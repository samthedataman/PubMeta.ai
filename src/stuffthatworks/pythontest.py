import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker

def classify_medical_text(input_text):
    # Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

    # Preprocess the input text
    tokenized_text = tokenizer.encode(input_text, truncation=True, padding=True)
    input_ids = torch.tensor([tokenized_text])

    # Perform text classification
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract predicted labels
    predicted_labels = torch.argmax(outputs.logits, dim=2)
    predicted_labels = predicted_labels.squeeze().tolist()

    # Create a reverse label map from label id to label string 
    reverse_label_map = {i: label for i, label in enumerate(model.config.id2label.values())}

    # Apply label map to the prediction
    predicted_labels_text = [reverse_label_map[i] for i in predicted_labels if reverse_label_map[i] != 'O']

    # Decode the tokens to get the original text (useful for checking and debugging)
    original_text = tokenizer.decode(tokenized_text)

    print("Predicted labels:", predicted_labels_text)
    print("Original medical text:", original_text)

    return predicted_labels_text,original_text


def classify_medical_text_insight(input_text):
    # Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

    # Preprocess the input text
    words = input_text.split()  # Split the text into words
    tokenized_inputs = tokenizer.encode_plus(words, truncation=True, padding=True, is_split_into_words=True, return_offsets_mapping=True, return_tensors='pt')
    input_ids = tokenized_inputs["input_ids"]

    # Perform text classification
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract predicted labels
    predicted_labels = torch.argmax(outputs.logits, dim=2)
    predicted_labels = predicted_labels.squeeze().tolist()

    # Create a reverse label map from label id to label string 
    reverse_label_map = {i: label for i, label in enumerate(model.config.id2label.values())}

    # Get original words and their corresponding predicted labels
    original_words_and_labels = []
    for offset_mapping, predicted_label in zip(tokenized_inputs["offset_mapping"].squeeze().tolist(), predicted_labels):
        # Ignore special tokens
        if offset_mapping[0] != 0 or offset_mapping[1] != 0:
            original_word = input_text[offset_mapping[0]:offset_mapping[1]]
            predicted_label_text = reverse_label_map[predicted_label]
            if predicted_label_text != 'O':
                original_words_and_labels.append((original_word, predicted_label_text))

    return original_words_and_labels

def concatenate_and_form_dictionary(data):
    concatenated_data = []
    for item in data:
        if isinstance(item, tuple):
            concatenated_data.extend(item)
        elif isinstance(item, list):
            concatenated_data += item

    dictionary = dict(zip(concatenated_data[::2], concatenated_data[1::2]))
    return dictionary

# # Example usage
# input_text = "Diabetes control blood sugar levels."
# predicted_labels = classify_medical_text(input_text)
# print(predicted_labels)

def fuzzy_match(df, column, search_string, threshold=50):
    # Load the ScispaCy model
    nlp = spacy.load("en_core_sci_sm")

    # Add the abbreviation pipe to the spacy pipeline.
    abbreviation_pipe = AbbreviationDetector(nlp)
    nlp.add_pipe(abbreviation_pipe)

    # Add the UMLS entity linker pipe to the spacy pipeline.
    linker = UmlsEntityLinker(resolve_abbreviations=True)
    nlp.add_pipe(linker)

    # tokenize the search string
    tokens = word_tokenize(search_string)

    # remove stopwords from the search string
    stopwords_removed = [word for word in tokens if not word in stopwords.words()]

    # join the words back together
    search_string = ' '.join(stopwords_removed)

    # process the text with the NLP model
    doc = nlp(search_string)

    # only keep recognized medical entities
    medical_terms = [str(ent) for ent in doc.ents]

    # get unique values from the column
    unique_values = df[column].unique()

    # find matches above the threshold
    matches = [value for value in unique_values for term in medical_terms if fuzz.ratio(term, value) > threshold]

    return matches



# Example usage
input_text = "Can you tell me the best diabetes treatment?"
predicted_labels = classify_medical_text_insight(input_text)
predicted_classes = classify_medical_text(input_text)

concatenate_and_form_dictionary(predicted_labels,predicted_classes)
