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
import re


def clean_reports_detrimental_percentage(value):
    try:
        match = re.search(r"\d+%", value)
        if match:
            return float(match.group().replace("%", ""))
        else:
            return ""
    except AttributeError:
        return ""


def clean_effectiveness_percentage(s):
    if s is None:
        return 0
    match = re.search(r"^(\d+)%", s)
    return int(match.group(1)) if match else 0


def clean_effectiveness_report(s):
    if s is None:
        return 0
    match = re.search(r"^(\d+)%", s)
    return s[match.end() :].strip() if match else s


def clean_brand_names(names):
    if names is None:
        return 0
    return names.replace("Brand names: ", "").split(",")


def clean_member_reports(reports):
    if reports is None:
        return 0
    return int(re.search(r"\d+", reports).group())


def clean_description(desc):
    # Split into sentences
    sentences = desc.split(". ")
    # Extract tried percentage, most tried rank and most effective rank
    tried_pct_match = re.search(r"\d+", sentences[1])
    tried_pct = int(tried_pct_match.group()) if tried_pct_match else 0
    most_tried_rank_match = re.search(r"\d+", sentences[2])
    most_tried_rank = int(most_tried_rank_match.group()) if most_tried_rank_match else 0
    most_effective_rank_match = re.search(r"\d+", sentences[3]) # this was previously incorrectly indexing sentences[2]
    most_effective_rank = int(most_effective_rank_match.group()) if most_effective_rank_match else 0
    return {
        "description": sentences[0],
        "tried_pct": tried_pct,
        "most_tried_rank": most_tried_rank,
        "most_effective_rank": most_effective_rank,
    }



def clean_rank(rank):
    if rank is None:
        return 0
    else:
        match = re.search(r"\d+", rank)
        if match:
            return int(match.group())
        else:
            return 0


def clean_other_treatment_counts(counts):
    if counts is None:
        return 0
    return int(re.search(r"\d+", counts).group())


def clean_reports_percentage(effectiveness_reports_percentage):
    lines = effectiveness_reports_percentage.split("\n")
    worked_extremely_well_pct = (
        int(re.search(r"\d+", lines[0]).group()) if re.search(r"\d+", lines[0]) else 0
    )
    worked_very_well_pct = (
        int(re.search(r"\d+", lines[1]).group())
        if (len(lines) > 1 and re.search(r"\d+", lines[1]))
        else 0
    )
    worked_fairly_well_pct = (
        int(re.search(r"\d+", lines[2]).group())
        if (len(lines) > 2 and re.search(r"\d+", lines[2]))
        else 0
    )
    non_significant_pct = (
        int(re.search(r"\d+", lines[3]).group())
        if (len(lines) > 3 and re.search(r"\d+", lines[3]))
        else 0
    )
    return [
        worked_extremely_well_pct,
        worked_very_well_pct,
        worked_fairly_well_pct,
        non_significant_pct,
    ]


def clean_reports_detrimental_percentage(reports):
    if reports is None:
        return 0

    try:
        return int(re.search(r"\d+", reports).group())
    except (AttributeError, ValueError):
        return 0


def clean_member_treatment_quotes(quotes):
    if quotes is None:
        return []
    return quotes.replace("Tap to contact\n", "").split("\n")


def clean_sideeffects(string):
    lines = string.split("\n")
    sideeffect = lines[0] if len(lines) > 0 else None
    percentage = None

    if len(lines) > 3:
        match = re.search(r"\d+", lines[3])
        if match is not None:
            percentage = int(match.group())

    return [sideeffect, percentage]


def clean_oftencombinedlist(list_):
    if list_ is None:
        return []

    lines = list_.split("\n")
    combined_list = []
    for i in range(0, len(lines), 2):
        try:
            drug = lines[i]
            reports = int(re.search(r"\d+", lines[i + 1]).group())
            combined_list.append({"drug": drug, "reports": reports})
        except AttributeError:
            # Skip iteration in case of StaleElementReferenceException
            continue
        except Exception as e:
            print(f"An error occurred: {e}")

    return combined_list


def insert_dataframe_into_table(df, frame_name):
    # Instantiate a client object using credentials
    project_name = "airflow-test-371320"
    dataset_name = "DEV"
    table_id = f"{dataset_name}.{frame_name}"
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


def get_conditions():
    # Instantiate a client object using credentials
    project_name = "airflow-test-371320"
    key_path = "/Users/samsavage/NHIB Scraper/airflow-test-371320-dad1bdc01d6f.json"
    creds = Credentials.from_service_account_file(key_path)
    client = bigquery.Client(credentials=creds, project=project_name)

    query = f"""
                SELECT *,
                IF(REGEXP_CONTAINS(Href, r'treatments/'), REGEXP_EXTRACT(Href, r'treatments/(.*)'), 
                REGEXP_EXTRACT(Href, r'([^/]*)$')) as treatment 
                FROM `airflow-test-371320.DEV.STUFF_THAT_WORKS_TREATMENTS_DEV_FULL_WITH_LINKS`
                where Href not in (

                SELECT distinct(Href) FROM `airflow-test-371320.DEV.SIDE_EFFECTS_TABLE_FULL`)
"""
    query_job = client.query(query)
    results = query_job.result().to_dataframe()
    sql_links = results["Href"].to_list()
    sql_links_length = len(sql_links)
    sql_conditions = results["conditions"].to_list()
    sql_treatments = results["treatment"].to_list()

    return sql_conditions, sql_links, sql_treatments, sql_links_length


def get_side_effects(link, condition, treatment):
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(
        "/Users/samsavage/Downloads/chromedriver_mac_arm64 (1)/chromedriver",
        options=chrome_options,
    )

    driver.get(link)

    scraped_data = {
        "condition": [condition],
        "treatment": [treatment],
        "brand_names": [],
        "member_reports": [],
        "drug_disease_description": [],
        "most_tried": [],
        "most_effective": [],
        "most_detrimental": [],
        "other_treatment_counts": [],
        "effectiveness_reports_percentage": [],
        "effectiveness_reports_detrimental_percentage": [],
        "member_treatment_quotes": [],
        "sideeffects": [],
        "oftencombinedlist": [],
        "Href": [link],
    }

    time.sleep(1)
    # Execute script to get brand names
    if driver.execute_script(
        "return document.querySelectorAll(\"[class*='treatment-title-brands']\")[0]"
    ):
        scraped_data["brand_names"] = [
            driver.execute_script(
                "return document.querySelectorAll(\"[class*='treatment-title-brands']\")[0]"
            ).get_attribute("innerText")
        ]
    else:
        scraped_data["brand_names"] = [""]

    # ...
    if driver.execute_script(
        "return document.querySelectorAll(\"[class*='report-number']\")[0]"
    ):
        scraped_data["member_reports"] = [
            driver.execute_script(
                "return document.querySelectorAll(\"[class*='report-number']\")[0]"
            ).get_attribute("innerText")
        ]
    else:
        scraped_data["member_reports"] = ""

    # ...
    if driver.execute_script(
        "return document.querySelectorAll(\"[class*='treatment-description']\")[0]"
    ):
        scraped_data["drug_disease_description"] = [
            driver.execute_script(
                "return document.querySelectorAll(\"[class*='treatment-description']\")[0]"
            ).get_attribute("innerText")
        ]
    else:
        scraped_data["drug_disease_description"] = [""]

    # ...
    # Similar for the other elements
    if driver.execute_script(
        "return document.querySelectorAll(\"[class*='entity-ranking-rank tried tried-with-detrimental']\")[0]"
    ):
        scraped_data["most_tried"] = [
            driver.execute_script(
                "return document.querySelectorAll(\"[class*='entity-ranking-rank tried tried-with-detrimental']\")[0]"
            ).get_attribute("innerText")
        ]
    else:
        scraped_data["most_tried"] = [""]

    print(scraped_data["most_tried"])

    if driver.execute_script(
        "return document.querySelectorAll(\"[class*='entity-ranking-rank effective']\")[0]"
    ):
        # Append most effective count to the dictionary
        scraped_data["most_effective"].append(
            driver.execute_script(
                "return document.querySelectorAll(\"[class*='entity-ranking-rank effective']\")[0]"
            ).get_attribute("innerText")
        )

    print(scraped_data["most_effective"])

    if driver.execute_script(
        "return document.querySelectorAll(\"[class*='entity-ranking-rank show-border-top']\")[0]"
    ):
        # Append most detrimental count to the dictionary
        scraped_data["most_detrimental"].append(
            driver.execute_script(
                "return document.querySelectorAll(\"[class*='entity-ranking-rank show-border-top']\")[0]"
            ).get_attribute("innerText")
        )
    else:
        scraped_data["most_detrimental"].append("")

    print(scraped_data["most_detrimental"])

    if driver.execute_script(
        "return document.querySelectorAll(\"[class*='stw-card cross-condition-entity-link']\")[0]"
    ):
        # Append other treatment counts to the dictionary
        scraped_data["other_treatment_counts"].append(
            driver.execute_script(
                "return document.querySelectorAll(\"[class*='stw-card cross-condition-entity-link']\")[0]"
            ).get_attribute("innerText")
        )
    else:
        scraped_data["other_treatment_counts"].append("")

    print(scraped_data["other_treatment_counts"])

    if driver.execute_script(
        "return document.querySelectorAll(\"[class*='treatment-effectiveness']\")[0]"
    ):
        # Append effectiveness reports percentage to the dictionary
        scraped_data["effectiveness_reports_percentage"].append(
            driver.execute_script(
                "return document.querySelectorAll(\"[class*='treatment-effectiveness']\")[0]"
            ).get_attribute("innerText")
        )
    else:
        scraped_data["effectiveness_reports_percentage"].append("")

    print(scraped_data["effectiveness_reports_percentage"])

    if driver.execute_script(
        "return document.querySelectorAll(\"[class*='treatment-effectiveness detrimental']\")[0]"
    ):
        # Append effectiveness reports detrimental percentage to the dictionary
        scraped_data["effectiveness_reports_detrimental_percentage"].append(
            driver.execute_script(
                "return document.querySelectorAll(\"[class*='treatment-effectiveness detrimental']\")[0]"
            ).get_attribute("innerText")
        )
    else:
        scraped_data["effectiveness_reports_detrimental_percentage"].append("")

    print(scraped_data["effectiveness_reports_detrimental_percentage"])

    if driver.execute_script(
        "return document.querySelectorAll(\"[class*='stw-card quotes-card']\")[0]"
    ):
        # Append member treatment quotes to the dictionary
        scraped_data["member_treatment_quotes"].append(
            driver.execute_script(
                "return document.querySelectorAll(\"[class*='stw-card quotes-card']\")[0]"
            ).get_attribute("innerText")
        )
    else:
        scraped_data["member_treatment_quotes"].append("")

    print(scraped_data["member_treatment_quotes"])

    print(f"scraping the hard stuff for {condition}")
    # Append side effects to the dictionary
    if driver.execute_script(
        "return document.querySelectorAll(\"[class*='show-more-button']\")[0]"
    ):
        print("displayed")
        click_counter = 0
        for i in range(0, 10):
            try:
                time.sleep(2)
                driver.execute_script(
                    "document.querySelectorAll(\"[class*='show-more-button']\")[0].click();"
                )
                click_counter += 1
                print(f"clicking {click_counter} times")
            except:
                continue

        if driver.execute_script(
            "return document.querySelectorAll(\"[class*='treatment-side-effect-row']\")"
        ):
            side_effects = driver.execute_script(
                "return Array.from(document.querySelectorAll(\"[class*='treatment-side-effect-row']\")).map(e => e.innerText)"
            )
            side_effects_str = "\n".join(side_effects)

            scraped_data["sideeffects"] = side_effects_str

            # scraped_data["sideeffects"].append(
            #     driver.execute_script(
            #         "return document.querySelectorAll(\"[class*='treatment-side-effect-row']\")[0]"
            #     ).get_attribute("innerText")
            # )
        else:
            scraped_data["sideeffects"].append("")

        # Append often combined list to the dictionary
        try:
            if driver.execute_script(
                "return document.querySelectorAll(\"[class*='View More']\")[0]"
            ).is_displayed():
                click_counter = 0
                for i in range(0, 10):
                    try:
                        time.sleep(2)
                        driver.execute_script(
                            "return document.querySelectorAll(\"[class*='View More']\")[0].click();"
                        )
                        click_counter += 1

                        print(f"clicking {click_counter} times")
                    except:
                        continue
        except:
            pass

            if driver.execute_script(
                "return document.querySelectorAll(\"[class*='combination']\")[0];"
            ):
                scraped_data["oftencombinedlist"].append(
                    driver.execute_script(
                        "return document.querySelectorAll(\"[class*='combination']\")[0]"
                    ).get_attribute("innerText")
                )
            else:
                scraped_data["oftencombinedlist"].append("")

        print(scraped_data)

        def pad_lists_in_dict(d):
            # Determine the maximum length of lists in dictionary values
            max_len = max(len(lst) for lst in d.values() if isinstance(lst, list))

            # Pad all lists to max_len with None
            for key, value in d.items():
                if isinstance(value, list):
                    d[key] = value + [None] * (max_len - len(value))

            return d

        scraped_data = pad_lists_in_dict(scraped_data)
        print(scraped_data)

        df = pd.DataFrame(scraped_data)
        df["brand_names"] = ",".join(
            df["brand_names"].apply(clean_brand_names).astype("str")
        )
        df["member_reports"] = df["member_reports"].apply(clean_member_reports)
        df["most_tried"] = df["most_tried"].apply(clean_rank)
        df["most_effective"] = df["most_effective"].apply(clean_rank)
        df["most_detrimental"] = df["most_detrimental"].apply(clean_rank)
        df["member_treatment_quotes"] = df["member_treatment_quotes"].str.replace(
            "Tap to contact", "|"
        )

        df["sideeffects"] = df["sideeffects"].astype(str)
        df["other_treatment_counts"] = df["other_treatment_counts"].apply(
            clean_other_treatment_counts
        )
        df["effectiveness_reports_detrimental_percentage"] = df[
            "effectiveness_reports_detrimental_percentage"
        ].apply(clean_reports_detrimental_percentage)

        return df


def main():
    try:
        sql_conditions, sql_links, sql_treatments, sql_links_length = get_conditions()

        dfs = []
        upload_counter = 0
        start_index = 0

        for i, (condition, link, treatment) in enumerate(
            zip(
                sql_conditions[start_index:],
                sql_links[start_index:],
                sql_treatments[start_index:],
            ),
            start=start_index,
        ):
            try:
                df = get_side_effects(link, condition, treatment)  # get the DataFrame
                dfs.append(df)  # add it to the list
                print(
                    f"Processing frame {i+1} out of next {min(100, sql_links_length - i)}."
                )

                # Every 10th DataFrame, concatenate and upload
                if (i + 1) % 100 == 0 or i == sql_links_length - 1:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    insert_dataframe_into_table(
                        combined_df, frame_name="STUFF_THAT_WORKS_SIDE_EFFECTS"
                    )
                    upload_counter += 1
                    # clear the list for the next batch of DataFrames
                    dfs = []
                    print(
                        f"Completed {upload_counter} upload(s) out of {sql_links_length // 100} expected uploads."
                    )

            except Exception as e:
                print(f"An error occurred: {e}")
                print("Resuming from the last successful frame...")
                start_index = i - ((i + 1) % 100) + 1

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
