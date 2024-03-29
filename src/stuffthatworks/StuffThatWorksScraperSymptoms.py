from selenium import webdriver
import time
import pandas as pd
import re
import datetime
from google.cloud import bigquery
from google.oauth2.service_account import Credentials
import pandas as pd
import pandas_gbq
from selenium.webdriver.chrome.options import Options


def get_symptoms(condition):
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(
        "/Users/samsavage/Downloads/chromedriver_mac_arm64 (1)/chromedriver",
        options=chrome_options,
    )

    value = condition
    driver.get(f"https://www.stuffthatworks.health/{value}/symptoms?tab=MostReported")
    print(f"found {value} most tried page scraping")
    # try:
    time.sleep(2)
    try:
        if driver.execute_script(
            "return document.querySelectorAll(\"[class*='more-result-button']\")[0];"
        ).is_displayed():
            click_counter = 0
            for i in range(0, 10):
                try:
                    time.sleep(2)
                    driver.execute_script(
                        "document.querySelectorAll(\"[class*='more-result-button']\")[0].click();"
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
            "return document.querySelectorAll(\"[class*='more-result-button']\")[1];"
        ).is_displayed():
            click_counter = 0
            for i in range(0, 10):
                try:
                    time.sleep(2)
                    driver.execute_script(
                        "document.querySelectorAll(\"[class*='more-result-button']\")[1].click();"
                    )
                    click_counter += 1
                    print(f"clicking {click_counter} times")
                except:
                    continue
        else:
            tiles = driver.execute_script(
                "return document.querySelectorAll(\"[class*='normalized-entity']\");"
            )
    except:
        print("no extra conditions")

    tiles = driver.execute_script(
        "return document.querySelectorAll(\"[class*='normalized-entity']\");"
    )
    time.sleep(2)

    print(f"length of tiles ======= {len(tiles)}")

    ranking_list = []
    symptoms_list = []
    num_reports_list = []
    condition_list = []

    pattern = r"^(#\d{1,2})([A-Za-z\s]+)(\d+)\s(reports)$"

    for tile in tiles:
        match = re.match(pattern, tile.get_attribute("textContent"))

        if match:
            ranking = match.group(1)
            condition = match.group(2)
            num_reports = match.group(3)

            print(f"Ranking: {ranking}")
            print(f"Condition: {condition}")
            print(f"Number of Reports: {num_reports}")
        else:
            print("No match found")
            ranking = None
            condition = None
            num_reports = None

        ranking_list.append(ranking)
        condition_list.append(value)
        symptoms_list.append(condition)
        num_reports_list.append(num_reports)

    df = pd.DataFrame(
        {
            "rankings": ranking_list,
            "symptoms": symptoms_list,
            "conditions": condition_list,
            "num_reports": num_reports_list,
            "TimeScraped": datetime.datetime.now(),
            "DateScraped": datetime.datetime.today().strftime("%m/%d/%Y"),
        }
    )
    return df


def insert_dataframe_into_table(df):
    # Instantiate a client object using credentials
    project_name = "airflow-test-371320"
    dataset_name = "DEV"
    table_id = f"{dataset_name}.STUFF_THAT_WORKS_SYMPTOMS_DEV_FULL"
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

    print(df.head().T)

    results = df["urlId"].unique()

    print(f"We are scraping for:{len(results)} conditions")
    not_exist_list = []

    for condition in results:
        try:
            counter_for_me = 0
            treatments_frame = get_symptoms(condition)
            counter_for_me += 1
            if len(treatments_frame) > 0:
                print(len(treatments_frame))
                insert_dataframe_into_table(treatments_frame)

                print(
                    f"We are ::::::{((counter_for_me/len(results))*100)}:::::completed"
                )
            else:
                print(f"frame is empty for {condition}")

                not_exist_list.append(condition)

                with open("exclusion_file.txt") as f:
                    f.write(not_exist_list)

            print(f"finish with {condition}!!!")
        except Exception as e:
            print(f"{condition} was {e} not valid moving on")
            continue


push_treatment_data_to_gbq()
