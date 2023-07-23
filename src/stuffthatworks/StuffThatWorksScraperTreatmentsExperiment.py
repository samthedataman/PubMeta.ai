import requests
from typing import Any, Dict, List, Optional, TypedDict
import pprint
from selenium import webdriver
import ast
import time
import pandas as pd
import re
import datetime
import json
from google.cloud import bigquery
from google.oauth2.service_account import Credentials
import pandas as pd
import openai
import os
import pandas_gbq
import streamlit as st
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service


def get_treatments(condition):
    print(f"this is the condition {condition}")
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(
        "/Users/samsavage/Downloads/chromedriver_mac_arm64 (1)/chromedriver",
        options=chrome_options,
    )

    driver.get(
        f"https://www.stuffthatworks.health/{condition}/treatments?tab=MostEffective"
    )

    print(f"found {condition} most tried page scraping")

    time.sleep(2)
    try:
        if driver.execute_script(
            "return document.querySelectorAll(\"[class*='view-more-wrapper']\")[0];"
        ).is_displayed():
            click_counter = 0
            for i in range(0, 10):
                try:
                    time.sleep(2)
                    driver.execute_script(
                        "document.querySelectorAll(\"[class*='view-more-wrapper']\")[0].click();"
                    )
                    click_counter += 1
                    print(f"clicking {click_counter} times")
                except:
                    continue
        else:
            print("click moving on to second clicker")
    except:
        pass
    ###################################
    try:
        if driver.execute_script(
            "return document.querySelectorAll(\"[class*='view-more-wrapper']\")[1];"
        ).is_displayed():
            click_counter = 0
            for i in range(0, 10):
                try:
                    time.sleep(2)
                    driver.execute_script(
                        "document.querySelectorAll(\"[class*='view-more-wrapper']\")[1].click();"
                    )
                    click_counter += 1
                    print(f"clicking {click_counter} times")
                except:
                    continue
        else:
            tile_list = []
            tiles = driver.execute_script(
                "return document.querySelectorAll(\"[class*='treatment-view-row']\");"
            )
    except:
        print("no extra conditions")

    tiles = driver.execute_script(
        "return document.querySelectorAll(\"[class*='treatment-view-row']\");"
    )
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    print(f"length of tiles ======= {len(tiles)}")
    ranking_list = []
    treatments_list = []
    num_reports_list = []
    conditions_list = []
    treatment_type_list = []

    pattern = r"^#(\d+)(.*?)(\d+)\s+reports(?:\s*(\d+)%?)?$"

    for tile in tiles:
        match = re.match(pattern, tile.get_attribute("textContent"))

        if match:
            ranking = match.group(1)
            treatments = match.group(2)
            num_reports = match.group(3)
            percentage = match.group(4)
            print(f"Ranking: {ranking}")
            print(f"Treatments: {treatments}")
            print(f"Number of Reports: {num_reports}")

            if percentage:
                good_or_bad_treatment = "Detrimental"
            else:
                good_or_bad_treatment = "Beneficial"
        else:
            print("No match found")
            ranking = None
            treatments = None
            num_reports = None
            good_or_bad_treatment = None

        treatment_type_list.append(good_or_bad_treatment)
        ranking_list.append(ranking)
        treatments_list.append(treatments)
        num_reports_list.append(num_reports)
        conditions_list.append(condition)

    df = pd.DataFrame(
        {
            "rankings": ranking_list,
            "treatments": treatments_list,
            "num_reports": num_reports_list,
            "conditions": conditions_list,
            "TreatmentType": treatment_type_list,
            "TimeScraped": datetime.datetime.now(),
            "DateScraped": datetime.datetime.today().strftime("%m/%d/%Y"),
        }
    )

    hrefs = []
    url_conditions = []
    # Find all 'a' tags
    time.sleep(2)
    elements = driver.find_elements(By.TAG_NAME, "a")

    # Loop through all 'a' tags
    for el in elements:
        href = el.get_attribute("href")
        hrefs.append(href)

    print(hrefs)

    # Remove None values from hrefs list
    hrefs = [href for href in hrefs if href is not None]
    desired_format = "https://www.stuffthatworks.health/"
    treatments_substring = "/treatments/"

    hrefs = [
        href
        for href in hrefs
        if href.startswith(desired_format) and treatments_substring in href
    ]
    print("this is the length of the links", len(hrefs))
    print("this is the length of the dataframe", len(df))
    # hrefs = [href for href in hrefs if condition.lower() in href.lower()]
    df["Href"] = hrefs
    # print(hrefs)
    # print(url_conditions)

    # print(f"The length or your {condition} list is = {len(hrefs)}")
    # print(f"The length or your {condition} list is = {len(url_conditions)}")
    # data = {"Href": hrefs, "Condition": url_conditions}
    # print(data)
    # dfList = pd.DataFrame(data)
    # print(dfList)
    # insert_dataframe_into_table_dict(dfList)

    return df


def insert_dataframe_into_table(df):
    # Instantiate a client object using credentials
    project_name = "airflow-test-371320"
    dataset_name = "DEV"
    table_id = f"{dataset_name}.STUFF_THAT_WORKS_TREATMENTS_DEV_FULL_WITH_LINKS"
    key_path = "/Users/samsavage/NHIB Scraper/airflow-test-371320-dad1bdc01d6f.json"
    creds = Credentials.from_service_account_file(key_path)
    client = bigquery.Client(credentials=creds, project=project_name)

    pandas_gbq.to_gbq(
        df,
        table_id,
        project_id="airflow-test-371320",
        if_exists="append",
        credentials=creds,
        chunksize=None,
    )


def insert_dataframe_into_table_dict(df):
    # Instantiate a client object using credentials
    project_name = "airflow-test-371320"
    dataset_name = "DEV"
    table_id = f"{dataset_name}.STUFF_THAT_WORKS_TREATMENTS_PAGES_LINKS_FULL_20_A_"
    key_path = "/Users/samsavage/NHIB Scraper/airflow-test-371320-dad1bdc01d6f.json"
    creds = Credentials.from_service_account_file(key_path)
    client = bigquery.Client(credentials=creds, project=project_name)

    pandas_gbq.to_gbq(
        df,
        table_id,
        project_id="airflow-test-371320",
        if_exists="append",
        credentials=creds,
        chunksize=None,
    )


def push_treatment_data_to_gbq():
    df = pd.read_csv("/Users/samsavage/PythonProjects/PubMedGPT/full_frame.csv")
    results = df["urlId"].unique()

    print(f"We are scraping for:{len(results)} conditions")

    for condition in results:
        print(condition)
        dfs_block = []
        counter_for_me = 0

        while counter_for_me < 2:
            try:
                treatments_frame_sub = get_treatments(condition)
                dfs_block.append(treatments_frame_sub)
                counter_for_me += 1
                print(counter_for_me)
            except:
                print(
                    f"could not fetch data for {condition}, moving to next condition."
                )
                break
        if counter_for_me >= 2:  # if we have 20 frames, then try to insert
            try:
                treatment_frame = pd.concat(dfs_block)
                insert_dataframe_into_table(treatment_frame)
                print("dictionary sent to google big query")
                print("frame inserted")
                print(
                    f"We are ::::::{((counter_for_me/len(results))*100)}:::::completed"
                )

                dfs_block = []  # Clear the list after successful insert
            except Exception as e:
                print(
                    f"Failed to insert data for {condition}, error was {e}, moving on"
                )
        else:
            print(f"Frame is empty for {condition}")

        print(f"Finished with {condition}!!!")


push_treatment_data_to_gbq()
