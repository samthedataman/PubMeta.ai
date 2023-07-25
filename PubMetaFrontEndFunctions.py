import pandas as pd
from fuzzywuzzy import fuzz
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from fuzzywuzzy import fuzz
import openai
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import Optional
from streamlit_chat import message
import openai
import json
import os
from fuzzywuzzy import fuzz


def fuzzy_match_user_query(
    user_search, diseases_list, treatments_list, score_cutoff=70
):
    # Prepare a combined dictionary for easier searching and categorization
    combined_list = {"Disease": diseases_list, "Treatment": treatments_list}
    result = {"Disease": [], "Treatment": []}

    for category, keyword_list in combined_list.items():
        for word in user_search.split():
            for keyword in keyword_list:
                score = fuzz.ratio(word.lower(), keyword.lower())
                if score >= score_cutoff:
                    result[category].append(keyword)

    return result


def def_old_chat():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_response(prompt):
        completions = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = completions.choices[0].text
        return message

    def get_text():
        input_text = st.text_input(
            "You: ", "Tell me what topic you would like to research", key="input"
        )
        return input_text

    with st.container():
        st.title("chatBot : Streamlit + openAI")

        if "generated" not in st.session_state:
            st.session_state["generated"] = []

        if "past" not in st.session_state:
            st.session_state["past"] = []

        user_input = get_text()

        if user_input:
            output = generate_response(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")


def set_custom_page_config():
    # st.set_page_config(
    #     page_title="PubMeta:The home to the most accurate chronic disease treatment analysis tool powered by user reports, pubmed.com, and Openai's GPT Large Langauge Model ",
    #     page_icon=":bar_chart:",
    #     layout="wide",
    # )

    st.markdown(
        """<link href="https://fonts.googleapis.com/css2?family=Major+Mono+Display&family=Xanh+Mono:ital@1&display=swap&family=Roboto:wght@100&display=swap&family=Roboto+Mono:wght@300&display=swap&family=Comfortaa:wght@400;500&family=Roboto+Mono:wght@300&display=swap" rel="stylesheet">""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<style>
            h1 { color: white; font-family: 'Comfortaa',monospace;} 
            h2 { color: white; font-family: 'Comfortaa', monospace;} 
            h3 { color: white; font-family: 'Comfortaa',monospace;} 
            .month { color: purple; margin-right:  }
            .blurb { color: white; font-family: 'Roboto Mono', sans-serif; line-height: 1.2 } 
            .item { display: flex; align-items: flex-start; } 
            .condition_title { color: gold; font-family: 'Comfortaa',sans-serif }
            .condition_links { color: gold; font-family: 'Comfortaa',sans-serif; line-height: 1.2 }
            </style>""",
        unsafe_allow_html=True,
    )
    st.markdown("""<style></style>""", unsafe_allow_html=True)

    st.markdown(
        """
    <style>
    .stWidget {
        background-color: teal;
    }
    .stContainer {
        background-color: gray;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


@st.cache_data
def fuzzy_match(df, symptoms_df, treatments_df, similarity_threshold=70):
    # Create an empty list to store the fuzzy matched words
    matched_words = []
    matched_words_treatments = []

    # Iterate over words in df1 and find the closest match in df2
    for word1 in df["ChronicCondition"].unique():
        best_match = None
        highest_similarity = 0

        for word2 in symptoms_df["conditions"].unique():
            similarity = fuzz.ratio(word1, word2)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = word2

        if highest_similarity >= similarity_threshold:
            matched_words.append((word1, best_match))

    # Create a DataFrame from the matched words
    matched_df = pd.DataFrame(
        matched_words, columns=["PubMedConditions", "StuffThatWorksConditions"]
    )

    # Iterate over words in df1 and find the closest match in df2
    for word1 in df["ChronicCondition"].unique():
        best_match = None
        highest_similarity = 0

        for word2 in treatments_df["conditions"].unique():
            similarity = fuzz.ratio(word1, word2)

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = word2

        if highest_similarity >= similarity_threshold:
            matched_words_treatments.append((word1, best_match))

    # Create a DataFrame from the matched words
    matched_df_treatment = pd.DataFrame(
        matched_words_treatments,
        columns=["PubMedConditions", "StuffThatWorksConditions"],
    )

    # Display the matched DataFrame
    df = df.merge(matched_df, left_on="ChronicCondition", right_on="PubMedConditions")

    symptoms_df = symptoms_df.merge(
        matched_df, left_on="conditions", right_on="StuffThatWorksConditions"
    )

    treatments_df = treatments_df.merge(
        matched_df_treatment, left_on="conditions", right_on="StuffThatWorksConditions"
    )

    return df, symptoms_df, treatments_df


def display_drugs_metrics(drug_list, drug_data):
    # Define the number of rows and columns based on the drug_list
    num_drugs = len(drug_list)
    metrics_per_row = min(3, num_drugs)  # Set the maximum columns per row
    num_containers = (num_drugs // metrics_per_row) + (
        num_drugs % metrics_per_row > 0
    )  # Round up

    drug_index = 0
    for container_index in range(num_containers):
        with st.container():
            cols = st.columns(metrics_per_row)
            for metric_index in range(metrics_per_row):
                if (
                    drug_index < num_drugs
                ):  # Check if there are still drugs left to display
                    drug_name = drug_list[drug_index]
                    drug_info = drug_data[drug_name]
                    rank = drug_info.get(
                        "people_rank", "N/A"
                    )  # Use "N/A" if "people_rank" is not present
                    with cols[metric_index]:
                        with st.expander(
                            str(f"Ranked #{rank} Overall by Patients"), expanded=True
                        ):
                            st.markdown(
                                f"<h3 style='text-align: center;'>{drug_name}</h3>",
                                unsafe_allow_html=True,
                            )
                            with st.expander("Studies"):
                                studies_count = drug_info["studies_count"]
                                st.metric(
                                    label="Number of Controled Trials (RCTs)",
                                    value=studies_count,
                                    delta=f"Studies in past 6 months: {rank}",
                                )
                                # Generate a small plot for the number of published papers over time
                                study_counts_over_time = drug_info[
                                    "study_counts_over_time"
                                ]
                                dates = [date for date, _ in study_counts_over_time]
                                counts = [count for _, count in study_counts_over_time]
                                fig = go.Figure(
                                    data=go.Scatter(
                                        x=dates,
                                        y=counts,
                                        mode="lines",
                                        name="studies over time",
                                    )
                                )
                                fig.update_layout(
                                    width=125,
                                    height=125,
                                    yaxis_type="log",  # Set y-axis scale to logarithmic
                                    margin=dict(
                                        l=20, r=20, t=30, b=20
                                    ),  # Adjust margins for a smaller plot
                                )
                                st.plotly_chart(fig)
                            with st.expander("Metrics"):
                                average_size = drug_info["average_size"]
                                general_trend = drug_info["general_trend"]
                                average_citation_count = drug_info[
                                    "average_citation_count"
                                ]
                                st.metric(
                                    label="Average Experiment Size ", value=average_size
                                )
                                st.metric(
                                    label="General Efficacy Trend", value=general_trend
                                )
                                st.metric(
                                    label="Average Citation Count",
                                    value=average_citation_count,
                                )
                                # Generate a small plot for the number of published papers over time
                            with st.expander("Best at improving:", expanded=False):
                                pathologies = drug_info["pathologies"]
                                st.write(pathologies)
                            with st.expander("Side Effects:", expanded=False):
                                side_effects = drug_info["side_effects"]
                                st.write(side_effects)
                            with st.expander("Full Studies:", expanded=False):
                                st.markdown(
                                    "Link to most Recent Study:", unsafe_allow_html=True
                                )
                                # In a new st.selectbox, create a dropdown for studies that is a clickable URL sorted by most cited
                                studies = sorted(
                                    drug_info["studies"],
                                    key=lambda x: x["citation_count"],
                                    reverse=True,
                                )
                                study_select = st.selectbox(
                                    f"Studies for {drug_name} (sorted by citation count)",
                                    options=[
                                        f"{study['date']}: {study['citation_count']} citations - {study['link']}"
                                        for study in studies
                                    ],
                                )
                                study_link = study_select.split(" - ")[-1]
                                st.markdown(
                                    f"[{study_select}]({study_link})",
                                    unsafe_allow_html=True,
                                )

                        drug_index += 1


@st.cache_data
def get_filtered_drug_list(
    # list selected by user (need to make one off)
    drug_list=None,
    drug_data=None,
    keyword=None,
    selected_symptoms=None,
    selected_diseases=None,
    selected_drugs=None,
    threshold=70,
):
    return [
        drug
        for drug in drug_list
        if any(
            symptom in drug_data[drug].get("Symptoms", [])
            for symptom in selected_symptoms
        )
        and any(
            disease in drug_data[drug].get("Pathologies", [])
            for disease in selected_diseases
        )
        and any(drug_name in drug.lower() for drug_name in selected_drugs)
        and (
            max(
                [
                    fuzz.ratio(keyword.lower(), symptom.lower())
                    for symptom in drug_data[drug].get("Symptoms", [])
                ]
            )
            >= threshold
            or max(
                [
                    fuzz.ratio(keyword.lower(), pathology.lower())
                    for pathology in drug_data[drug].get("Pathologies", [])
                ]
            )
            >= threshold
            or max(
                [
                    fuzz.ratio(keyword.lower(), study.lower())
                    for study in drug_data[drug].get("Studies", [])
                ]
            )
            >= threshold
            or fuzz.ratio(keyword.lower(), drug.lower()) >= threshold
        )
    ]


class TopicList(BaseModel):
    Disease: Optional[str] = Field(None, description="The name of the disease")
    Treatment: Optional[str] = Field(None, description="The treatment for the disease")


def fuzzymatchuserinput(prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    user_input = prompt
    completions = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"""
        Can you output the diseases, conditions, or treatments listed in the response by a user: {user_input}

        Rules: Only output diseases, conditions, or treatments if they are present in the text provided by the user!
        Examples of treatments could include vitamins, specific medications, execerices,lifestyle change, drugs, physical or mental therapy and other practices to remedy a disease or condition
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
    parser = PydanticOutputParser(pydantic_object=TopicList)

    parsed_output = parser.parse(message)

    return parsed_output.Disease, parsed_output.Treatment


def fuzzymatchresponse(prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    user_input = prompt
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"""
        Can you output the diseases, conditions, or treatments listed in the response by a user: {user_input}

        Rules: Only output diseases, conditions, or treatments if they are present in the text provided by the user!
        Examples of treatments could include vitamins, specific medications, execerices,lifestyle change, drugs, physical or mental therapy and other practices to remedy a disease or condition
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
            "cancer"
            "Inflammatory bowel disease" 

            ^^there are alot more disease please your knowledge base to make this determination

        FORMAT CONTEXT: Output via a dictionary that looks like this:

        EXAMPLE RESPONSE : '{{"Disease":"Ankylosing Spondylitis","Treatment":"Vitamin D"}}'
        """,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.0,
    )
    message = completions.choices[0].text.strip()  # remove leading/trailing whitespace

    try:
        message_dict = json.loads(message)  # Parse the message into a Python dictionary
        parsed_output = TopicList(
            **message_dict
        )  # Validate and parse into the Pydantic object
    except (json.JSONDecodeError, ValidationError):
        return "Sorry, I can't find a condition or treatment in your query."

    return parsed_output
