import streamlit as st
from streamlit_chat import message
import pandas as pd
from langchain.llms import OpenAI
import os
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
import plotly.express
from streamlit_searchbox import st_searchbox
from typing import List, Tuple

from src.stuffthatworks.StuffThatWorksETL import run_jobs
from google.cloud import bigquery
from google.oauth2 import service_account
from langchain.chains import ConversationalRetrievalChain
import streamlit_nested_layout
from langchain.vectorstores import Chroma
from langchain import PromptTemplate

import json
from pydantic import ValidationError
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.memory import ConversationBufferMemory
import math
import pandas as pd
from PubMetaAppBackEndFunctions import *
from chatbotfunctions import *
import pandas as pd
import streamlit as st
import openai
from pydantic import BaseModel, Field
from typing import Optional
from streamlit_chat import message
import openai
from fuzzywuzzy import fuzz
from langchain.prompts.chat import SystemMessagePromptTemplate
from dotenv import load_dotenv
import os
import langchain

# load .env file
load_dotenv()

# from dotenv import load_dotenv

st.set_page_config(
    page_title="PubMeta.ai",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="auto",
)

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# function to search diseases


import time
from collections import defaultdict

# list of all diseases


def build_substring_dict(diseases: list):
    substring_dict = {}

    for disease in diseases:
        for i in range(len(disease)):
            for j in range(i + 1, len(disease) + 1):
                substring = disease[i:j]
                if substring not in substring_dict:
                    substring_dict[substring] = set()
                substring_dict[substring].add(disease)

    return substring_dict


def search_diseases_dict_optimized(substring_dict: dict, searchterm: str):
    searchterm = searchterm.lower()
    if searchterm in substring_dict:
        return list(substring_dict[searchterm])
    else:
        return []

    # List of diseases, this can be fetched from your data source
diseases = [
        "gallstones",
        "intracranial-hypertension",
        "laryngopharyngeal-reflux",
        "lipoedema",
        "migraine",
        "osteoarthritis",
        "schizoaffective-disorder",
        "sibo",
        "barretts-esophagus",
        "copd",
        "crohns-disease-in-adults",
        "lupus",
        "post-traumatic-stress-disorder-ptsd",
        "raynauds-syndrome",
        "shingles",
        "eye-floaters",
        "lichen-planus",
        "long-term-effects-of-covid-19",
        "myasthenia-gravis",
        "adhd-children-teens",
        "dissociative-identity-disorder",
        "hypothyroidism",
        "lyme-disease",
        "macular-degeneration",
        "male-erectile-dysfunction",
        "overactive-bladder",
        "rheumatoid-arthritis",
        "epilepsy",
        "menopause",
        "pots",
        "scoliosis",
        "adhd-adults",
        "anemia",
        "hernias",
        "panic-disorder",
        "urinary-tract-infection",
        "essential-tremor",
        "atrial-tachyarrhythmias",
        "chronic-knee-pain",
        "diverticulosis",
        "gad-teens",
        "pmdd",
        "ptsd-and-cptsd",
        "tourette-syndrome",
        "osteoporosis",
        "tmj",
        "asthma-in-teens",
        "polymyalgia-rheumatica",
        "reflux",
        "seborrheic-dermatitis",
        "epstein-barr-virus-ebv",
        "genital-herpes",
        "vertigo-unspecified",
        "atopic-dermatitis-in-adults",
        "ankylosing-spondylitis",
        "chronic-pain",
        "endometriosis",
        "heart-failure",
        "hypertension",
        "interstitial-cystitis",
        "lactose-intolerance",
        "myalgic-encephalomyelitis",
        "pancreatitis",
        "tension-headache",
        "fibromyalgia",
        "mixed-depressive-anxiety",
        "parosmia",
        "bronchiectasis",
        "chronic-constipation",
        "clinical-depression-in-seniors",
        "lymphocytic-colitis",
        "peripheral-neuropathy",
        "secondary-progressive-ms-spms",
        "clinical-depression",
        "complex-post-traumatic-stress-disorder-c-ptsd",
        "gad-adults",
        "hyperthyroidism",
        "mortons-neuroma",
        "parkinsons-disease",
        "sleep-apnea",
        "small-fiber-neuropathy",
        "occipital-neuralgia",
        "herniated-disc",
        "dysthymia",
        "multiple-sclerosis",
        "spinal-stenosis",
        "bipolar-type-1-disorder",
        "mcas",
        "psoriasis",
        "fnd",
        "low-back-pain",
        "restless-legs-syndrome",
        "acne",
        "arfid",
        "pcos",
        "social-anxiety",
        "asthma-in-seniors",
        "chronic-kidney-disease",
        "chronic-urticaria",
        "cluster-headache",
        "crohns-disease",
        "degenerative-disc-disease",
        "fibroids",
        "hidradenitis-suppurativa",
        "lymphoedema",
        "borderline-personality",
        "thyroiditis-non-hashimotos",
        "binge-eating-disorder",
        "high-cholesterol",
        "rosacea",
        "clinical-depression-in-teens",
        "diverticulitis",
        "gout",
        "asthma-in-adults",
        "bipolar-disorder",
        "bulimia-nervosa",
        "celiac",
        "hsd",
        "hyperhidrosis-excessive-sweating",
        "mctd",
        "type-2-diabetes",
        "anorexia-nervosa-restricting-type",
        "clinical-depression-in-adults",
        "irritable-bowel-syndrome",
        "microscopic-colitis",
        "plantar-fasciitis",
        "sinus-problems",
        "type-1-diabetes-lada",
        "autism-spectrum-disorder",
        "bipolar-type-2-disorder",
        "perimenopause",
        "psoriatic-arthritis",
        "schizophrenia",
        "anorexia-nervosa",
        "crps",
        "insomnia",
        "nafld",
        "new-daily-persistent-headache-ndph",
        "menieres-disease",
        "natural-menopause",
        "perioral-dermatitis",
        "sjogrens-syndrome",
        "tinnitus",
        "bells-palsy",
        "trigeminal-neuralgia",
        "anorexia-nervosa-binge-eatingpurge-type",
        "burning-mouth-syndrome",
        "ocd",
        "asthma",
        "clinical-depression-in-young-adults",
        "gastroparesis",
        "human-papillomavirus",
        "lichen-sclerosus",
        "morgellons",
        "chronic-lyme-disease-cld",
        "recurrent-bacterial-vaginosis",
        "ulcerative-colitis",
        "adrenal-insufficiency",
        "atopic-eczema",
        "dyshidrotic-eczema",
    ]


# Build the substring dictionary from the list of diseases
substring_dict = build_substring_dict(diseases)

# Now, you can use the search_diseases_dict_optimized function in your search box like this:


# Search function takes disease map as input
def search_diseases_optimized(searchterm, disease_map):
    results = disease_map.get(searchterm.lower(), [])
    results = results[:5]

    return results


# @st.cache_data
def search_diseases(searchterm: str):
    # diseases = get_unique_diseases()
    diseases = [
        "gallstones",
        "intracranial-hypertension",
        "laryngopharyngeal-reflux",
        "lipoedema",
        "migraine",
        "osteoarthritis",
        "schizoaffective-disorder",
        "sibo",
        "barretts-esophagus",
        "copd",
        "crohns-disease-in-adults",
        "lupus",
        "post-traumatic-stress-disorder-ptsd",
        "raynauds-syndrome",
        "shingles",
        "eye-floaters",
        "lichen-planus",
        "long-term-effects-of-covid-19",
        "myasthenia-gravis",
        "adhd-children-teens",
        "dissociative-identity-disorder",
        "hypothyroidism",
        "lyme-disease",
        "macular-degeneration",
        "male-erectile-dysfunction",
        "overactive-bladder",
        "rheumatoid-arthritis",
        "epilepsy",
        "menopause",
        "pots",
        "scoliosis",
        "adhd-adults",
        "anemia",
        "hernias",
        "panic-disorder",
        "urinary-tract-infection",
        "essential-tremor",
        "atrial-tachyarrhythmias",
        "chronic-knee-pain",
        "diverticulosis",
        "gad-teens",
        "pmdd",
        "ptsd-and-cptsd",
        "tourette-syndrome",
        "osteoporosis",
        "tmj",
        "asthma-in-teens",
        "polymyalgia-rheumatica",
        "reflux",
        "seborrheic-dermatitis",
        "epstein-barr-virus-ebv",
        "genital-herpes",
        "vertigo-unspecified",
        "atopic-dermatitis-in-adults",
        "ankylosing-spondylitis",
        "chronic-pain",
        "endometriosis",
        "heart-failure",
        "hypertension",
        "interstitial-cystitis",
        "lactose-intolerance",
        "myalgic-encephalomyelitis",
        "pancreatitis",
        "tension-headache",
        "fibromyalgia",
        "mixed-depressive-anxiety",
        "parosmia",
        "bronchiectasis",
        "chronic-constipation",
        "clinical-depression-in-seniors",
        "lymphocytic-colitis",
        "peripheral-neuropathy",
        "secondary-progressive-ms-spms",
        "clinical-depression",
        "complex-post-traumatic-stress-disorder-c-ptsd",
        "gad-adults",
        "hyperthyroidism",
        "mortons-neuroma",
        "parkinsons-disease",
        "sleep-apnea",
        "small-fiber-neuropathy",
        "occipital-neuralgia",
        "herniated-disc",
        "dysthymia",
        "multiple-sclerosis",
        "spinal-stenosis",
        "bipolar-type-1-disorder",
        "mcas",
        "psoriasis",
        "fnd",
        "low-back-pain",
        "restless-legs-syndrome",
        "acne",
        "arfid",
        "pcos",
        "social-anxiety",
        "asthma-in-seniors",
        "chronic-kidney-disease",
        "chronic-urticaria",
        "cluster-headache",
        "crohns-disease",
        "degenerative-disc-disease",
        "fibroids",
        "hidradenitis-suppurativa",
        "lymphoedema",
        "borderline-personality",
        "thyroiditis-non-hashimotos",
        "binge-eating-disorder",
        "high-cholesterol",
        "rosacea",
        "clinical-depression-in-teens",
        "diverticulitis",
        "gout",
        "asthma-in-adults",
        "bipolar-disorder",
        "bulimia-nervosa",
        "celiac",
        "hsd",
        "hyperhidrosis-excessive-sweating",
        "mctd",
        "type-2-diabetes",
        "anorexia-nervosa-restricting-type",
        "clinical-depression-in-adults",
        "irritable-bowel-syndrome",
        "microscopic-colitis",
        "plantar-fasciitis",
        "sinus-problems",
        "type-1-diabetes-lada",
        "autism-spectrum-disorder",
        "bipolar-type-2-disorder",
        "perimenopause",
        "psoriatic-arthritis",
        "schizophrenia",
        "anorexia-nervosa",
        "crps",
        "insomnia",
        "nafld",
        "new-daily-persistent-headache-ndph",
        "menieres-disease",
        "natural-menopause",
        "perioral-dermatitis",
        "sjogrens-syndrome",
        "tinnitus",
        "bells-palsy",
        "trigeminal-neuralgia",
        "anorexia-nervosa-binge-eatingpurge-type",
        "burning-mouth-syndrome",
        "ocd",
        "asthma",
        "clinical-depression-in-young-adults",
        "gastroparesis",
        "human-papillomavirus",
        "lichen-sclerosus",
        "morgellons",
        "chronic-lyme-disease-cld",
        "recurrent-bacterial-vaginosis",
        "ulcerative-colitis",
        "adrenal-insufficiency",
        "atopic-eczema",
        "dyshidrotic-eczema",
    ]

    # filter diseases based on the search term
    return [d for d in diseases if searchterm.lower() in d.lower()]


# @st.cache_data(ttl=400)
def get_vbd():
    embeddings = OpenAIEmbeddings()
    vector_db = load_faiss_from_gcs("pubmeta", "index", embeddings=embeddings)
    return embeddings, vector_db


def set_css(css: str):
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def set_bot_css():
    css = """
    .chatbot {
        font-size: 20px; 
    }
    """
    set_css(css)


set_bot_css()


# @st.cache_data(experimental_allow_widgets=True)
def chat_bot_streamlit_openai():
    st.header("Pick a New Condition to get started!🚀")

    full_user_question = ""
    search_response = ""

    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    if "memory" not in st.session_state:
        st.session_state["memory"] = []

    if "reset_input" not in st.session_state:
        st.session_state["reset_input"] = False

    col1, col2 = st.columns(2)

    with col1:
        # input_disease = st_searchbox(
        #     label="↳Pick a New Condition",
        #     search_function=search_db(),
        #     default=["ankylosing-spondylitis"],
        # )

        input_disease = st_searchbox(
            lambda searchterm: search_diseases_dict_optimized(
                substring_dict, searchterm
            ),
            "Search a New Condition (this may take one second or two)...",
            key="disease_searchbox",
            label="↳Pick a Condition to Research!",
            default="ankylosing-spondylitis",
        )

        # Usage
        # input_disease = st_searchbox(
        #     lambda term: search_diseases_optimized(term, disease_map),
        #     "Search conditions",
        #     key="disease_searchbox",
        # )

        # st.write(input_disease)

        # input_disease = "".join([i for i in input_disease])

        if not input_disease:
            input_disease = ""

        if "input_disease" not in st.session_state:
            st.session_state.input_disease = False

        if input_disease or st.session_state.input_disease:
            st.session_state.input_disease = True
    with col2:
        drop_down_options = st.selectbox(
            "↳Pick a Research Topic Chat Injection",
            options=[
                "🏥 Compare Treatment Benefits",
                "🩹 Compare Treatment Side Effects",
                "📝 Compare Treatment Member Reports",
                "🚨 Compare Treatment Triggers",
                "🤕 Compare Treatment Comorbities",
                "📚 Compare Treatment Studies",
                "📚 Most-Cited-Study",
                "📈 Popular-Treatment-Report",
                "📊 Database-Knowledge-Enumeration",
                "💊 Detailed-Treatment-Information",
                "🏥 Detailed-Disease-Information",
                "🔍 Specific-Study-Insights",
                "🌐 General-Disease-Treatment-Overview",
                "📝 User-Report-Summary",
                "🆕 New-Treatment-Options",
                "📈 Statistically-Significant-Treatments",
                "📝 User-Intensive-Treatment-Options",
                "🕰️ Prognosis-Information",
                "⚠️ Side-Effects-Information",
                "👤 Personalized-Treatment-Information",
                "📑 Treatment-Procedure-Details",
                "📈 Disease-Progression-Information",
                "💪 Lifestyle-Modification-Suggestions",
                "🧬 Hereditary-Risk-Insights",
                "🔬 Diagnostic-Tests-Details",
                "🛡️ Disease-Prevention-Strategies",
                "💉 Vaccine-Information",
                "🌿 Complementary-Therapies-Insights",
                "👴 Age-Related-Risks-Information",
                "👫 Gender-Specific-Information",
                "⚠️ Disease-specific-Risk-Factors",
                "🔬 Experimental-Treatments-Insights",
            ],
            index=5,
        )

    input_treatment_type = st.sidebar.selectbox(
        f"↳View Beneficial OR Detrimental Treatments",
        ["Beneficial", "Detrimental"],
        key="treatment_type",
        index=0,
    )

    if not input_treatment_type:
        input_treatment_type = ""

    if "input_treatment_type" not in st.session_state:
        st.session_state.input_treatment_type = False

    input_treatment = st.sidebar.multiselect(
        f"↳Treatment Compare Tool",
        get_treatments_for_diseases(input_disease, input_treatment_type),
        key="treatment_sidebar",
    )
    if not input_treatment:
        input_treatment = ""

    if "input_treatment" not in st.session_state:
        st.session_state.input_treatment = False

    if input_treatment or st.session_state.input_disease:
        st.session_state.input_treatment = True
    else:
        input_treatment = ""

    # if input_disease:

    #     symptoms_df, triggers_df, comorbidities_df, treatments_df = run_jobs()

    #     symp_frame_viz_frame = treatments_df[
    #         (treatments_df["conditions"] == str(input_disease[0]))
    #         & (treatments_df["TreatmentType"] == "Beneficial")
    #     ]
    #     symp_frame_viz_frame["treatments"] = (
    #         symp_frame_viz_frame["treatments"].str.split(",").str[0]
    #     )
    #     # Create the stacked bar chart
    #     fig = px.bar(
    #         symp_frame_viz_frame.sort_values(by="num_reports", ascending=False).head(5),
    #         x="treatments",
    #         y="num_reports",
    #         color="treatments",
    #         title=f"Top Treatments for {str(input_disease[0])}",
    #         labels={
    #             "treatments": "Treatments",
    #             "num_reports": "Number of Reports",
    #         },
    #         height=500,
    #     )
    #     fig.update_layout(showlegend=False)

    # Display the chart using Streamlit

    # if len(input_disease) > 0:
    #     st.markdown(
    #         f"""
    #         <h2 style="color: blue;">Compare treatments for chronic conditions side-by-side using AI and the latest medical research</h2>
    #         <h2>Researching <span style="color: orange;">{input_disease[0]}</span></h2>
    #         """,
    #         unsafe_allow_html=True,
    #     )
    # else:
    #     st.subheader(
    #         f"""Compare treatments for chronic conditions side-by-side using AI and the latest medical research
    #         Researching {input_disease}"""
    #     )

    # col1, col2 = st.columns(2)

    with st.expander("Want to talk to PubMeta.ai?", expanded=True):
        if (st.session_state.input_disease) or (
            st.session_state.input_disease and st.session_state.input_treatment
        ):
            if "full_user_question" not in st.session_state:
                st.session_state.full_user_question = False

            if full_user_question or st.session_state.input_disease:
                st.session_state.full_user_question = True

            # st.sidebar.plotly_chart(fig, use_container_width=True)

    if input_treatment:
        default_text = (
            ""
            if st.session_state["reset_input"]
            else f"Hello, can you research {drop_down_options} for {input_disease} combined with treatments such as : {' vs '.join(input_treatment)}"
        )
        full_user_question = st.text_input(
            "Chat with me!",
            default_text,
            key="full_user_question_key_when_using_tabs",
        )
    else:
        default_text = (
            ""
            if st.session_state["reset_input"]
            else f"Hello, can you research {drop_down_options} for {input_disease}"
        )
        full_user_question = st.text_input(
            "Chat with me!",
            default_text,
            key="full_user_question_key_when_using_tabs",
        )

    enter_button = st.button("Click to chat with PubMeta")

    # st.balloons()

    ###Customer Journey 1 Step 2: They have used drop downs and now are searching for the data/answers from the chat bot
    if ((input_disease and input_treatment) or (input_disease)) and enter_button:
        # get query based on user input
        embeddings, vector_db = get_vbd()

        df = get_disease_by_treatment_data(
            input_disease, input_treatment, input_treatment_type
        )
        # get similar results from db

        search_response, search_history_outchain = retreive_best_answer(
            full_user_question, embeddings, vector_db
        )
        # for i in range(100):
        # # Increment progress bar
        #     progress_bar.progress(i+1)
        #     time.sleep(0.01)

        # Clear progress bar
        # store the output
        st.session_state.past.append(full_user_question)
        st.session_state.generated.append(search_response)
        st.session_state.memory.append(search_history_outchain)

        # st.subheader(f"Question:",full_user_question)

        # st.write(search_response)
        # st.write(st.session_state["memory"][-1])

        # if "first_run" not in st.session_state:
        #     st.session_state["first_run"] = True
        #     message("Hello! I'm your chatbot. How can I assist you today?")

        # if st.session_state["generated"]:
        #     for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        #         message(st.session_state["generated"][i], key=str(i))

        #         message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")

        # if not input_disease or input_treatment:
        #     parsed_output = fuzzy_match_with_query(
        #         full_user_question,
        #         get_unique_diseases(),
        #         get_unique_treatment(),
        #         score_cutoff=58,
        #     )
    if input_disease:
        st.subheader(f"Top Treatments for :orange[{str(input_disease)}]")
    # else:
    #     st.subheader("Pick a Condition above to start your analysis")
    panel_df = get_disease_by_treatment_data(
        input_disease, input_treatment, input_treatment_type
    )

    display_treatments_metrics(
        panel_df, input_disease, input_treatment_type, input_treatment
    )

    pass


# Start timer
start_time = time.time()

# Track number of signups
num_signups = 0


chat_bot_streamlit_openai()
