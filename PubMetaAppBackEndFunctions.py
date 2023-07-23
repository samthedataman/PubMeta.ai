import pandas as pd
from pymed import PubMed
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import openai
from collections import OrderedDict
from habanero import counts
import regex as re
from google.cloud import bigquery
from google.oauth2.service_account import Credentials
from google.oauth2 import service_account
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import time
import streamlit as st
import math
import plotly.express as px

from dotenv import load_dotenv
import os

# load .env file
load_dotenv()
# load_dotenv()


def convert_user_question_to_dict(prompt):
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
    user_input = prompt
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"""
        Can you output the diseases, conditions, or treatments listed in the response by a user: {user_input}

        Rules: Only output diseases, conditions, or treatments if they are present in the text provided by the user!
        Examples of treatments could include vitamins, specific medications, execerices, drugs, physical or mental therapy and other practices to remedy a disease
        Examples of conditions include :  "Heart disease",
            "Diabetes",
            "Arthritis",
            "Asthma",
            "Chronic kidney disease",
            "Chronic obstructive pulmonary disease (COPD)",
            "Cystic fibrosis",
            "Hypertension (High Blood Pressure)",
            "Parkinson's disease",
            "Multiple sclerosis",
            "Alzheimer's disease",
            "Osteoporosis",
            "Epilepsy",
            "HIV/AIDS",
            "Lupus",
            "Inflammatory bowel disease" 

        FORMAT CONTEXT: Output via a dictionary that looks like this:

        EXAMPLE RESPONSE : '{{"Disease":"Ankylosing Spondylitis","Treatment":"Vitamin D"}}'
        """,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.0,
    )
    message = completions.choices[0].text
    return message


def classify_medical_text_insight(input_text):
    # Load pre-trained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

    # Preprocess the input text
    words = input_text.split()  # Split the text into words
    tokenized_inputs = tokenizer.encode_plus(
        words,
        truncation=True,
        padding=True,
        is_split_into_words=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = tokenized_inputs["input_ids"]

    # Perform text classification
    with torch.no_grad():
        outputs = model(input_ids)

    # Extract predicted labels
    predicted_labels = torch.argmax(outputs.logits, dim=2)
    predicted_labels = predicted_labels.squeeze().tolist()

    # Create a reverse label map from label id to label string
    reverse_label_map = {
        i: label for i, label in enumerate(model.config.id2label.values())
    }

    # Get original words and their corresponding predicted labels
    original_words_and_labels = []
    for offset_mapping, predicted_label in zip(
        tokenized_inputs["offset_mapping"].squeeze().tolist(), predicted_labels
    ):
        # Ignore special tokens
        if offset_mapping[0] != 0 or offset_mapping[1] != 0:
            original_word = input_text[offset_mapping[0] : offset_mapping[1]]
            predicted_label_text = reverse_label_map[predicted_label]
            if predicted_label_text != "O":
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
    reverse_label_map = {
        i: label for i, label in enumerate(model.config.id2label.values())
    }

    # Apply label map to the prediction
    predicted_labels_text = [
        reverse_label_map[i] for i in predicted_labels if reverse_label_map[i] != "O"
    ]

    # Decode the tokens to get the original text (useful for checking and debugging)
    original_text = tokenizer.decode(tokenized_text)

    print("Predicted labels:", predicted_labels_text)
    print("Original medical text:", original_text)

    return predicted_labels_text


def stringify_columns(df):
    """
    Convert all columns in a DataFrame to strings, with special handling for dictionaries.

    :param df: pandas DataFrame
    :return: DataFrame with all columns converted to strings
    """
    for column in df.columns:
        df[column] = df[column].apply(
            lambda x: ", ".join(f"{k}:{v}" for k, v in x.items())
            if isinstance(x, dict)
            else str(x)
        )
    return df


def get_condition_treatments_for_pubmed_from_STW(query):
    # Instantiate a client object using credentials
    project_name = "airflow-test-371320"
    key_path = "/Users/samsavage/PythonProjects/PubMedGPT/data/gcp_creds.json"
    creds = Credentials.from_service_account_file(key_path)

    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = bigquery.Client(credentials=credentials, project=project_name)

    query = query
    query_job = client.query(query)
    results = query_job.result().to_dataframe()

    Disease = [d for d in results["DiseaseTreatmentKey"].unique()]

    # Saving to JSON
    with open("DiseaseTreatments.json", "w") as f:
        json.dump(Disease, f)

    return Disease


def handle_list_objects(series):
    # Convert list objects to bytes
    series = series.apply(
        lambda x: json.dumps(x).encode() if isinstance(x, list) else x
    )
    return series


def upload_to_bq(df):
    project_name = "airflow-test-371320"
    dataset_name = "PubMeta"
    table_id = f"{dataset_name}.Articles_PubMed_July_2nd_base"

    creds = service_account.Credentials.from_service_account_file(
        "/Users/samsavage/PythonProjects/PubMedGPT/data/gcp_creds.json"
    )

    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    # GoogleCloudBaseHook
    df.to_gbq(
        table_id,
        if_exists="append",
        project_id=project_name,
        credentials=credentials,
    )
    print("data loaded to db")


def update_dicts(article_dicts):
    for article_dict in article_dicts:
        # Extract PMID and add 'ArticlePmid' and 'ArticleLink' keys to the dict
        pmid_search = re.search(r"\d+", article_dict.get("pubmed_id", ""))
        if pmid_search:
            pmid = pmid_search.group()
            article_dict["ArticlePmid"] = pmid
            article_dict["ArticleLink"] = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
        # Add 'CitationCounts' key to the dict
        citation_count = 0  # Default to 0 if we can't get a count
        doi = article_dict.get("doi", "")
        if doi:
            doi = doi.split(" ")[0]  # get the first DOI
            try:
                citation_count = counts.citation_count(doi=doi)
            except Exception:
                # If there's an error getting the citation count, we'll just ignore it.
                pass
        article_dict["CitationCounts"] = citation_count
    return article_dicts


def get_ai_generated_data(df):
    prompt = PromptTemplate(
        input_variables=["title", "results", "conclusions", "abstract"],
        template="""Context:
                    You are a medical PhD student at Harvard reviewing PubMed medical journal articles to perform a meta analysis.
                    You are reviewing a paper titled: {title}
                    The paper had these conclusions: {conclusions}
                    The paper had these results: {results}
                    The paper has this abstract: {abstract}

                    Task: Return 6 labels for each paper

                    1) Study Objective OR Hypothesis
                    2) Outcomes Measured
                    3) Treatment efficacy for each outcome tested, only choose from these 4 options below:
                    (Statistically significant high (context: if p value mentioned less than .05))
                    (Statistically significant middle (context: if p value mentioned less than .1))
                    (Statistically significant low (context: if p value mentioned less than .2))
                    (Directionally significant (context: if p value is not mentioned but there is an effect))
                    (No effect (context: results were inconclusive))
                    3) Sample size of study (number of total participants)
                    4) Study type (meta, randomized, clinical, double blind, qualitative)
                    5) Stat sig + / 1 with disease specific symptoms addressed by study (specific metrics relative to the disease) example = (.05+, medical acronym/indicator)
                    6) Trend/correlation discovered

                    Format requirements:
                    Return a Python Dictionary {}. Here is an example:
                    {
                    "Study Objective OR Hypothesis": "Examining the effectiveness of telephone cognitive behavior therapy (CBT) for chronic fatigue syndrome (CFS)", 
                    "Outcomes Measured": "Improvement in physical functioning and reduction in fatigue", 
                    "Treatment Efficacy": "Statistically significant high (p<.05)", 
                    "Sample Size": "30", 
                    "Study Type": "Randomized clinical trial", 
                    "Stat sig + / 1 with Disease Specific Symptoms Addressed by Study": [".01+, time elapsed before recovery of 25% of neuromuscular function", ".001+, time elapsed before recovery of 90% of function"], 
                    "Trend/Correlation Discovered": "Rocuronium block had a better profile than vecuronium block for patients with myasthenia gravis"
                    }
                    """,
    )

    llm = OpenAI(
        model_name="text-davinci-003",  # default model
        temperature=0.0,  # temperature dictates how whacky the output should be
    )

    llmchain = LLMChain(llm=llm, prompt=prompt)
    print(llmchain)

    df["MetaGPT"] = df.apply(
        lambda row: llmchain.run(
            title=row["title"],
            results=row["results"],
            conclusions=row["conclusions"],
            abstract=row["abstract"],
        ),
        axis=1,
    )

    # df["MetaGPT"] = df["MetaGPT"].apply(ast.literal_eval)

    return df


def get_results_and_predictions(DiseaseTreatments):
    # Iterate over the list
    print(f"length of Dtreatment list is {len(DiseaseTreatments)}, scraping away..")

    dt_count = 0
    for item in DiseaseTreatments:
        print(f"scraping {item}")
        try:
            # Split the element into disease and treatment
            disease, treatment = item.split("|")
        except ValueError:
            print(
                f"Skipping item '{item}' because it can't be split into disease and treatment."
            )
            continue

        print(f"Scraping {disease} with {treatment}")

        # Clean up the strings (remove leading/trailing spaces)
        disease = disease.strip()
        treatment = treatment.strip()

        # Build the query
        query = f"({disease}[Title/Abstract] AND {treatment}[Title/Abstract])"
        #
        pubmed = PubMed(tool="MyTool", email="razoranalyticsconsulting@gmail.com")
        my_api_key = "c7cc7e152fff1679a4d539bd35b270803f08"
        pubmed.parameters.update({"api_key": my_api_key})
        pubmed._rateLimit = 45

        time.sleep(3)
        # Execute the query
        try:
            results = pubmed.query(query=query, max_results=250)
            article_dicts = []
            for article in results:
                # Convert article to dict and append to list
                article_dict = article.toDict()
                # article_dict.pop("authors", None)
                # Combine disease and treatment as 'DiseaseTreatments'
                DiseaseTreatments = disease + "|" + treatment
                # Create an ordered dictionary with 'DiseaseTreatments' as the first key
                ordered_dict = OrderedDict(
                    [("DiseaseTreatments", DiseaseTreatments)]
                    + list(article_dict.items())
                )
                ordered_dict["disease"] = disease
                ordered_dict["treatment"] = treatment
                article_dicts.append(ordered_dict)
                article_dicts = update_dicts(article_dicts)
        except:
            print(f"{item} does not have any RCT's at the moment")

        df = pd.DataFrame(article_dicts)
        df = stringify_columns(df)
        print(f"uploading {disease} {treatment} pubmed results")
        dt_count += 1
        print(df.head(10).T)

        if len(df) > 0:
            # funtion to get_gen
            # df = get_ai_generated_data(df)
            try:
                upload_to_bq(df)
                print(dt_count)
                print(
                    f"done with {round(dt_count/len(DiseaseTreatments),3)}% of  pubmed results"
                )
            except:
                print("table schema issue skipping load")
                continue

        else:
            print("no date to scrape")

    return df
