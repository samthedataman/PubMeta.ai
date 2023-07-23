from selenium import webdriver
import time
import pandas as pd
import re
from google.cloud import bigquery
from google.oauth2.service_account import Credentials
import pandas as pd
import pandas_gbq
from selenium.webdriver.chrome.options import Options
import time
from pandas_gbq.exceptions import GenericGBQException
import re
from google.cloud import bigquery
from google.oauth2.service_account import Credentials
from google.api_core.exceptions import Forbidden
import pandas_gbq


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
    tried_pct = int(re.search(r"\d+", sentences[1]).group())
    most_tried_rank = int(re.search(r"\d+", sentences[2]).group())
    most_effective_rank = int(re.search(r"\d+", sentences[2]).group())
    return {
        "description": sentences[0],
        "tried_pct": tried_pct,
        "most_tried_rank": most_tried_rank,
        "most_effective_rank": most_effective_rank,
    }


def extract_ranks(description):
    most_tried_rank = re.search(r"Ranked #(\d+) most tried", description)
    most_effective_rank = re.search(r"#(\d+) most effective", description)

    if most_tried_rank and most_effective_rank:
        return {
            "most_tried_rank": int(most_tried_rank.group(1)),
            "most_effective_rank": int(most_effective_rank.group(1)),
        }
    else:
        return {
            "most_tried_rank": None,
            "most_effective_rank": None,
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
    # Split into lines and extract drug and report count as list of dictionaries
    lines = list_.split("\n")
    combined_list = [
        {"drug": lines[i], "reports": int(re.search(r"\d+", lines[i + 1]).group())}
        for i in range(0, len(lines), 2)
    ]
    return combined_list


def get_conditions():
    # Instantiate a client object using credentials
    project_name = "airflow-test-371320"
    key_path = "/Users/samsavage/NHIB Scraper/airflow-test-371320-dad1bdc01d6f.json"
    creds = Credentials.from_service_account_file(key_path)
    client = bigquery.Client(credentials=creds, project=project_name)

    # WHERE Href NOT IN (
    #   SELECT DISTINCT(Href)
    #   FROM `airflow-test-371320.DEV.SIDE_EFFECTS_TABLE_FULL`
    query = f"""SELECT *,
  IF(REGEXP_CONTAINS(Href, r'treatments/'), REGEXP_EXTRACT(Href, r'treatments/(.*)'), 
  REGEXP_EXTRACT(Href, r'([^/]*)$')) as treatment 
FROM `airflow-test-371320.DEV.STUFF_THAT_WORKS_TREATMENTS_DEV_FULL_WITH_LINKS`
LIMIT 2000 OFFSET 4000;"""

    query_job = client.query(query)
    results = query_job.result().to_dataframe()
    sql_links = results["Href"]
    sql_conditions = results["conditions"]
    sql_treatments = results["treatment"]

    return sql_conditions, sql_links, sql_treatments


def insert_dataframe_into_table(df):
    primary_table_id = "SIDE_EFFECTS_TABLE_BATCH_PROD_GODZILLA_MODE_OG"
    # Instantiate a client object using credentials
    project_name = "airflow-test-371320"
    dataset_name = "DEV"
    key_path = "/Users/samsavage/NHIB Scraper/airflow-test-371320-dad1bdc01d6f.json"
    creds = Credentials.from_service_account_file(key_path)
    client = bigquery.Client(credentials=creds, project=project_name)

    # Prepare the full ID for the primary table
    primary_full_table_id = f"{dataset_name}.{primary_table_id}"

    try:
        pandas_gbq.to_gbq(
            df,
            primary_full_table_id,
            project_id="airflow-test-371320",
            if_exists="append",
            credentials=creds,
            chunksize=None,
        )
    except Forbidden as e:
        if "quotaExceeded" in str(e):
            print(f"Quota exceeded for {primary_full_table_id}")


def pad_lists_in_dict(d):
    # Determine the maximum length of lists in dictionary values
    max_len = max(len(lst) for lst in d.values() if isinstance(lst, list))

    # Pad all lists to max_len with None
    for key, value in d.items():
        if isinstance(value, list):
            d[key] = value + [None] * (max_len - len(value))

    return d


def clean_columns(df):
    try:
        df["brand_names"] = ",".join(
            df["brand_names"].apply(clean_brand_names).astype("str")
        )
    except Exception as e:
        print(f"Error while processing 'brand_names': {e}")
        df["brand_names"] = None

    try:
        df["member_reports"] = df["member_reports"].apply(clean_member_reports)
    except Exception as e:
        print(f"Error while processing 'member_reports': {e}")
        df["member_reports"] = 0

    # try:
    #     df["most_tried"] = df["most_tried"].apply(clean_rank)
    # except Exception as e:
    #     print(f"Error while processing 'most_tried': {e}")
    #     df["most_tried"] = 0

    # try:
    #     df["most_effective"] = df["most_effective"].apply(clean_rank)
    # except Exception as e:
    #     print(f"Error while processing 'most_effective': {e}")
    #     df["most_effective"] = 0

    try:
        df["most_detrimental"] = df["most_detrimental"].apply(clean_rank)
    except Exception as e:
        print(f"Error while processing 'most_detrimental': {e}")
        df["most_detrimental"] = 0

    try:
        df["member_treatment_quotes"] = df["member_treatment_quotes"].str.replace(
            "Tap to contact", "|"
        )
    except Exception as e:
        print(f"Error while processing 'member_treatment_quotes': {e}")
        df["member_treatment_quotes"] = 0

    try:
        df["sideeffects"] = df["sideeffects"].astype(str)
    except Exception as e:
        print(f"Error while processing 'sideeffects': {e}")
        df["sideeffects"] = 0

    try:
        df["other_treatment_counts"] = df["other_treatment_counts"].apply(
            clean_other_treatment_counts
        )
    except Exception as e:
        print(f"Error while processing 'other_treatment_counts': {e}")
        df["other_treatment_counts"] = 0

    try:
        df["effectiveness_reports_detrimental_percentage"] = df[
            "effectiveness_reports_detrimental_percentage"
        ].apply(clean_reports_detrimental_percentage)
    except Exception as e:
        print(
            f"Error while processing 'effectiveness_reports_detrimental_percentage': {e}"
        )
        df["effectiveness_reports_detrimental_percentage"] = 0

    return df


def main():
    # example usage
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(
        "/Users/samsavage/Downloads/chromedriver_mac_arm64 (1)/chromedriver",
        options=chrome_options,
    )
    sql_conditions, sql_links, sql_treatments = get_conditions()
    counter = 0
    df_counter = 0
    upload_counter = 0
    dfs = []

    for condition, link, treatment in zip(sql_conditions, sql_links, sql_treatments):
        driver.get(link)
        print(f"FOR {(condition)} ON {treatment} retreiving {link}")
        counter += 1
        scraped_data = {
            "condition": [condition],
            "treatment": [treatment],
            "brand_names": [],
            "member_reports": [],
            "drug_disease_description": [],
            "most_tried_rank": [],
            "most_effective_rank": [],
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
            description = driver.execute_script(
                "return document.querySelectorAll(\"[class*='treatment-description']\")[0]"
            ).get_attribute("innerText")
            scraped_data["drug_disease_description"] = [description]
            ranks = extract_ranks(description)
            scraped_data.update(ranks)
        else:
            scraped_data["drug_disease_description"] = [""]
            scraped_data["most_tried_rank"] = 0
            scraped_data["most_effective_rank"] = 0

        # ...
        # Similar for the other elements
        # if driver.execute_script(
        #     "return document.querySelectorAll(\"[class*='entity-ranking-rank tried tried-with-detrimental']\")[0]"
        # ):
        #     scraped_data["most_tried"] = [
        #         driver.execute_script(
        #             "return document.querySelectorAll(\"[class*='entity-ranking-rank tried tried-with-detrimental']\")[0]"
        #         ).get_attribute("innerText")
        #     ]
        # else:
        #     scraped_data["most_tried"] = [0]

        # # print(scraped_data["most_tried"])

        # if driver.execute_script(
        #     "return document.querySelectorAll(\"[class*='entity-ranking-rank effective']\")[0]"
        # ):
        #     # Append most effective count to the dictionary
        #     scraped_data["most_effective"].append(
        #         driver.execute_script(
        #             "return document.querySelectorAll(\"[class*='entity-ranking-rank effective']\")[0]"
        #         ).get_attribute("innerText")
        #     )

        # print(scraped_data["most_effective"])

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
            scraped_data["most_detrimental"].append(0)

        # print(scraped_data["most_detrimental"])

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
            scraped_data["other_treatment_counts"].append(0)

        # print(scraped_data["other_treatment_counts"])

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
            scraped_data["effectiveness_reports_percentage"].append(0)

        # print(scraped_data["effectiveness_reports_percentage"])

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
            scraped_data["effectiveness_reports_detrimental_percentage"].append(0)

        # print(scraped_data["effectiveness_reports_detrimental_percentage"])

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

        # print(scraped_data["member_treatment_quotes"])

        # print(f"scraping the hard stuff for {condition}")
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
                    print(
                        "An exception occurred, but continuing with the same iteration."
                    )

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
                        print(
                            "An exception occurred, but continuing with the same iteration."
                        )

        except:
            print("An exception occurred, but continuing with the same iteration.")

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

        scraped_data = pad_lists_in_dict(scraped_data)

        df = pd.DataFrame(scraped_data)

        df = clean_columns(df)

        dfs.append(df)
        df_counter += 1
        print(
            f" we have a total of {df_counter} frames so far in batch # {upload_counter}  out of {len(sql_links)}"
        )

        print(f"df_counter: {df_counter}")  # Added for debugging

        if df_counter % 50 == 0:
            print("We are joining list and serving to database")
            # Concatenate all dataframes in the batch and insert them
            batch_df = pd.concat(dfs, ignore_index=True)
            print("data types of batch before conversion")
            print(batch_df.dtypes)  # Check data types before conversion

            print(batch_df.shape)
            print(batch_df.head(10).T)

            print(
                "About to call insert_dataframe_into_table()..."
            )  # Added for debugging
            try:
                object_columns = ['condition', 'treatment', 'brand_names', 'drug_disease_description',
                                'effectiveness_reports_percentage', 'member_treatment_quotes',
                                'sideeffects', 'oftencombinedlist', 'Href']

                batch_df[object_columns] = batch_df[object_columns].astype(str)
                
                int64_columns = ['member_reports', 'most_tried_rank', 'most_effective_rank',
                                'most_detrimental', 'other_treatment_counts',
                                'effectiveness_reports_detrimental_percentage']

                batch_df[int64_columns] = batch_df[int64_columns].fillna(0).astype(int)



                print("data types of batch after conversion")
                print(batch_df.dtypes)  # Check data types before conversion
                insert_dataframe_into_table(batch_df)
                upload_counter += 1
                print(
                    "Successfully called insert_dataframe_into_table()"
                )  # Added for debugging

                print("uploading batched frame")
                print(
                    f"We have uploaded {len(batch_df)*upload_counter} tables out of {len(sql_links)} or {round(len(batch_df)*upload_counter/len(sql_links), 6)}"
                )
                # Clear the batch and print progress
                dfs = []
                df_counter = 0  # Reset the counter after each batch upload
            except Exception as e:
                print(f"Error while processing large dataframe: {e}")

    # Don't forget to insert the last batch if it's not empty and has fewer than 1000 dataframes
    if dfs:
        batch_df = pd.concat(dfs, ignore_index=True)
        insert_dataframe_into_table(batch_df)


main()
