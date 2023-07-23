import pandas as pd
from typing import List
from streamlit_searchbox import st_searchbox

# Assuming you have pulled data from BigQuery into a DataFrame 'df'
# with columns 'disease' and 'treatment'
# df = ...


def search_dataframe(searchterm: str) -> List[any]:
    if searchterm:
        mask = df.apply(
            lambda row: searchterm.lower() in str(row["disease"]).lower()
            or searchterm.lower() in str(row["treatment"]).lower(),
            axis=1,
        )
        return df[mask].to_dict("records")
    else:
        return []


import streamlit as st


def app():
    st.title("Disease and Treatment Autocomplete Search")

    # Pass search function to searchbox
    selected_value = st_searchbox(
        search_function=search_dataframe,
        placeholder="Search diseases or treatments...",
        label="Autocomplete Searchbox",
        key="dt_searchbox",
    )

    if selected_value:
        st.write(f"You have selected: {selected_value}")
    else:
        st.write("No selection made.")


if __name__ == "__main__":
    app()
