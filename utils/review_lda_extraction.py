import pandas as pd
from tqdm import tqdm
from gensim.models import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from nltk import word_tokenize, pos_tag

from clean_category import *


# Nouns are more useful under current scinerio, take only nouns
def nouns(text):
    """apply on cleaned text
    return a list of extracted nouns"""

    is_noun = lambda pos: pos[:2] == "NN"
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
    return all_nouns


def lemma_noun(words):
    """
    apply on a list of nouns
    return a list of lemmatized nouns
    """
    from nltk.stem import WordNetLemmatizer

    Lem = WordNetLemmatizer()
    lemword = list(map(lambda word: Lem.lemmatize(word), words))
    return lemword


def lemma_noun_df(df: object, col_org, col_final):
    """
    apply on df, return a df that has a col with name col_final consisted with lists of nouns
    """
    df[col_final] = df[col_org].apply(lambda x: lemma_noun(nouns(x)))
    return df


if __name__ == "__main__":
    # Read data
    df_review_raw = pd.read_csv(
        "data/data_processed/matched_review_with_class.csv",
        index_col=False,
        dtype=object,
    )
    df_web_and_class = pd.read_csv(
        "data/data_processed/matched_scrapped_carpe.csv", index_col=False
    )
    business_set = df_web_and_class["company_id"].unique()
    # Here we only take business of interests
    # businesses that have matching naics code from carpe and scrapped file
    df_review_index = (
        df_review_raw["company_id"]
        .apply(lambda x: x if x in business_set else None)
        .dropna()
        .index
    )
    df_review = df_review_raw.iloc[df_review_index]

    # Takeout all of the none values in review content
    df_review_has_value = df_review[df_review["content"] != None]
    df_review_has_value = df_review_has_value[
        df_review_has_value["content"] != "novalue"
    ]
    df_review_has_value = df_review_has_value[df_review_has_value["content"] != "None"]
    df_review_has_value = df_review_has_value[df_review_has_value["content"].notna()]

    # data preprocessing
    # Group by company_id and join all of the reviews into one document
    # Apply data cleaning as well
    review = pd.DataFrame(
        df_review_has_value.groupby("company_id")["content"].apply(" ".join)
    ).reset_index()
    review_clean = data_clean(review, "content", "content_clean")
    review_clean["content_clean_list"] = review_clean["content_clean"].apply(
        lambda x: word_tokenize(x)
    )
    review_clean = lemma_noun_df(review_clean, "content_clean", "content_clean_noun")

    # Apply LDA
    topic_lda = []
    for i, c in tqdm(review_clean.iterrows(), total=review_clean.shape[0]):
        try:
            # Put words into dictionary and tokenize to prepare for LDA
            dictionary = Dictionary([c["content_clean_noun"]])
            corpus = [dictionary.doc2bow(text) for text in [c["content_clean_noun"]]]
            # If noun is less than 5 (not enough words): use the complete list
            if len(dictionary) <= 5:
                dictionary = Dictionary([c["content_clean_list"]])
                corpus = [
                    dictionary.doc2bow(text) for text in [c["content_clean_list"]]
                ]
            lda = LdaMulticore(corpus, num_topics=1)
            # Take top 10 topics
            topic = [t[0] for t in lda.get_topic_terms(0, 10)]
            topic_words = [dictionary[t] for t in topic]
            topic_lda.append(topic_words)
        except Exception as e:
            # if exception exists: append None
            topic_lda.append(None)

    # merge data
    review_clean["topic_lda"] = topic_lda
    df_web_and_class_copy = df_web_and_class.merge(
        review_clean[["company_id", "topic_lda"]], how="outer"
    ).fillna("novalue")
    df_web_and_class_copy.to_pickle(
        "data/data_processed/web_scrapped_carpe_with_review_topic.pickle"
    )
