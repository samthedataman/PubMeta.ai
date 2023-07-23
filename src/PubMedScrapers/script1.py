from PubMetaAppBackEndFunctions import (
    get_condition_treatments_for_pubmed_from_STW,
    get_results_and_predictions,
)


def main():
    DiseaseTreatments = get_condition_treatments_for_pubmed_from_STW(
        query="""SELECT * 
    FROM `airflow-test-371320.PubMeta.PubMetaArticlesFull` 
    WHERE TreatmentCategory IN ('Drug', 'Coping tools and strategies', 'Lifestyle change')
    AND 
    DiseaseTreatmentKey NOT IN (
    SELECT distinct DiseaseTreatments
FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY ArticleLink, ArticlePmid, DiseaseTreatments ORDER BY DiseaseTreatments) AS row_num
  FROM (
    SELECT *,"" as isbn,"" as langauge,"" as publication_type,"" as sections,"" AS publisher, "" as publisher_location FROM `airflow-test-371320.PubMeta.ArticlesM1CHIPALL`
    UNION ALL
    SELECT *,"" as isbn,"" as langauge,"" as publication_type,"" as sections,"" AS publisher, "" as publisher_location FROM `airflow-test-371320.PubMeta.ArticlesM1CHIP`
    UNION ALL
    SELECT *,"" as isbn,"" as langauge,"" as publication_type,"" as sections,"" AS publisher, "" as publisher_location FROM `airflow-test-371320.PubMeta.ArticlesM1CHIPa`
    UNION ALL 
    SELECT * FROM airflow-test-371320.PubMeta.Articles_PubMed_July_2nd_base
    UNION ALL 
    SELECT *,"" as isbn,"" as langauge,"" as publication_type,"" as sections,"" AS publisher, "" as publisher_location FROM `airflow-test-371320.PubMeta.Articles_PubMed_July_1st`
  ) AS combined_data
)
WHERE row_num = 1)
    
    
    
    ;
    """
    )
    print(DiseaseTreatments[0:10])

    get_results_and_predictions(DiseaseTreatments)
    print("Data upload process completed.")


if __name__ == "__main__":
    main()
