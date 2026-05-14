import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from rake_nltk import Rake
import gensim.downloader as api

from clean_category import *

m = api.load("glove-twitter-200")


def classify(
    naics,
    category,
    index,
    naics_col=None,
    cat_col="category_clean",
    weight_col=None,
    embed=False,
    per_naics=False,
    peak=False,
    top_n_pairs=5,
):
    """
    Wrapper function for all kinds of classification, 4 in total
    """
    return classify_embed_per_naics(
        naics,
        category,
        index,
        cat_col=cat_col,
        naics_col=naics_col,
        top_n_pairs=top_n_pairs,
    )


if __name__ == "__main__":
    df_web_and_class = pd.read_pickle(
        "data/data_processed/web_scrapped_carpe_with_review_topic.pickle"
    )
    # Remove rows with no categories
    df_web_and_class = df_web_and_class[df_web_and_class["categories"] != "[]"]

    category = df_web_and_class.copy()
    # convert to list and clean the category column
    category["topic_lda"] = category["topic_lda"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    category = clean_class_df(category, "categories")
    category = remove_stopwords_df(category, "category_clean")
    category = category.groupby("company_id").first().reset_index()
    category = lemmatize_class(category, col="category_clean")
    category = category[category.category_clean.str.len() > 0].reset_index()

    # apply rake to get topics from webpage and description
    rake_business_web_topic = []
    stopword = stopwords.words("english")
    for i, c in category.iterrows():
        topic_rake = []
        if c["webpage"] != "novalue":
            r = Rake(max_length=4)
            r.extract_keywords_from_text(c["webpage"])
            topic_rake.extend(r.get_ranked_phrases()[0:10])
        if c["description"] != "novalue":
            r = Rake(stopwords=stopword, max_length=4)
            r.extract_keywords_from_text(c["description"])
            topic_rake.extend(r.get_ranked_phrases()[0:10])
            rake_business_web_topic.append(topic_rake)
        else:
            rake_business_web_topic.append([])
    category["topic_rake"] = rake_business_web_topic

    # combine lda and rake topics
    cat_combined = []
    for i, c in category.iterrows():
        temp = []
        if c["topic_rake"] != []:
            temp.extend(c["topic_rake"])
        if c["topic_lda"] != []:
            temp.extend(c["topic_lda"])
        cat_combined.append(temp)
    category["topic_combined"] = cat_combined

    # embed_xxx means all word vectors in xxx are add together and averaged to create a single vector
    category["embed_clean"] = embed_all(category, embed_col="category_clean")
    category["embed_lda"] = embed_all(category, embed_col="topic_lda")
    category["embed_rake"] = embed_all(category, embed_col="topic_rake")
    category["embed_combined"] = embed_all(category, embed_col="topic_combined")

    # If there is no topics to use, all weights are 1
    # Weight using each topic words (one vector per word)
    category["weight_lda"] = weight_all(category, column="topic_lda")
    category["weight_rake"] = weight_all(category, column="topic_rake")
    category["weight_combined"] = weight_all(category, column="topic_combined")

    # Weight topics using category
    category["lda_weight"] = weight_all(
        category, column="category_clean", embed_col="topic_lda"
    )
    category["rake_weight"] = weight_all(
        category, column="category_clean", embed_col="topic_rake"
    )
    category["combined_weight"] = weight_all(
        category, column="category_clean", embed_col="topic_combined"
    )

    category["category_rake_rounded"] = round_top_n_cat(
        category, "weight_rake", "rake_weight", "topic_rake", "category_clean", n=6
    )
    category["category_combined_rounded"] = round_top_n_cat(
        category,
        "weight_combined",
        "combined_weight",
        "topic_combined",
        "category_clean",
        n=6,
    )
    category["embed_rake_rounded"] = embed_all(
        category, embed_col="category_rake_rounded"
    )
    category["embed_combined_rounded"] = embed_all(
        category, embed_col="category_combined_rounded"
    )

    category = category[category.category_combined_rounded.str.len() == 6].reset_index(
        drop=True
    )
    print(category.shape)
    print("Business Data Preparation is Finished, Begin Naics Data Preparation")

    naics_2dig = pd.read_csv(
        "data/naics/2-digit.csv", sep="\t", header=0, index_col=False, dtype=object
    )
    naics_4dig = pd.read_csv(
        "data/naics/4-digit.csv", sep="\t", header=0, index_col=False, dtype=object
    )
    naics_6dig = pd.read_csv(
        "data/naics/6-digit.csv", sep="\t", header=0, index_col=False, dtype=object
    )

    naics2 = clean_desc_df(naics_2dig, "Title")
    naics4 = clean_desc_df(naics_4dig, "Title")
    naics6 = clean_desc_df(naics_6dig, "Title")
    # Some extra steps to clean the naics titles
    stopword.extend(
        [
            "sector",
            "whole",
            "establishments",
            "service",
            "services",
            "office",
            "offices",
            "company",
            "establishment",
            "establishments",
            "engaged",
            "primarily",
            "activity",
            "activities",
            "facilities",
            "facility",
            "include",
            "includes",
            "industry",
            "industries",
            "provide",
            "provides",
            "specialize",
            "example",
            "includes",
            "include",
            "subsector",
            "other",
            "may",
            "operator",
            "often",
            "center",
            "centers",
        ]
    )

    naics4 = remove_stopwords_title(naics4, "Title", stopwords=stopword)
    naics6 = remove_stopwords_title(naics6, "Title", stopwords=stopword)

    # This is for 2 digit naics code only
    # Here I defined a few ways to extract the "topic"
    # first one is Rake
    # second is 4 digit title
    # thrid is 6 digit title
    # fourth is 4 digit and 6 digit title combined
    topic_naics_4_dig_title = []
    topic_naics_6_dig_title = []

    for i, c in naics2.iterrows():
        topic_4_dig_title = []
        topic_6_dig_title = []

        # Get titles from 4 digit and 6 digit naics code
        temp_code = c["Code"]
        if temp_code in set(["31-33", "44-45", "48-49"]):
            temp_code_list = temp_code.split("-")
            if temp_code_list[0] == "31":
                temp_code_list.append("32")
            temp_code_set = set(temp_code_list)
        else:
            temp_code_set = set([temp_code])
        for j, c2 in naics4.iterrows():
            if c2["Code"][0:2] in temp_code_set:
                topic_4_dig_title.append(c2["Title"])

        for k, c3 in naics6.iterrows():
            if c3["Code"][0:2] in temp_code_set:
                topic_6_dig_title.append(c3["Title"])

        topic_naics_4_dig_title.append(list(set(topic_4_dig_title)))
        topic_naics_6_dig_title.append(list(set(topic_6_dig_title)))

    naics2["desc_topic_4_dig_title"] = topic_naics_4_dig_title
    naics2["desc_topic_6_dig_title"] = topic_naics_6_dig_title

    print("Starting Experiment, testing top-K pairs")

    for test_n in range(1, 16):
        print("Working on top " + str(test_n) + " pairs")
        classification_6dig_title_embed = []
        classification_6dig_title_embed_rake_rounded = []
        classification_6dig_title_embed_combined_rounded = []

        for i, c in category.iterrows():
            # classify with different topic modeling result
            # [:,0] here means takes only the naics number

            c3 = classify(
                naics2,
                category,
                i,
                naics_col="desc_topic_6_dig_title",
                cat_col="embed_clean",
                embed=True,
                per_naics=True,
                top_n_pairs=test_n,
            )[:, 0]
            c4 = classify(
                naics2,
                category,
                i,
                naics_col="desc_topic_6_dig_title",
                cat_col="embed_rake_rounded",
                embed=True,
                per_naics=True,
                top_n_pairs=test_n,
            )[:, 0]
            c5 = classify(
                naics2,
                category,
                i,
                naics_col="desc_topic_6_dig_title",
                cat_col="embed_combined_rounded",
                embed=True,
                per_naics=True,
                top_n_pairs=test_n,
            )[:, 0]

            classification_6dig_title_embed.append(c3)
            classification_6dig_title_embed_rake_rounded.append(c4)
            classification_6dig_title_embed_combined_rounded.append(c5)

        category["classification_6dig_title_embed_{}".format(test_n)] = (
            classification_6dig_title_embed
        )
        category["classification_6dig_title_embed_rake_rounded_{}".format(test_n)] = (
            classification_6dig_title_embed_rake_rounded
        )
        category[
            "classification_6dig_title_embed_combined_rounded_{}".format(test_n)
        ] = classification_6dig_title_embed_combined_rounded

        category.to_pickle("category_top_n_pairs.pickle")

    print("Experiment done, preserving test result")
    category.to_pickle("category_top_n_pairs.pickle")
