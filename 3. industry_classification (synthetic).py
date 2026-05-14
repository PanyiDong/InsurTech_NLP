import re
import os
import ast
import argparse
import warnings
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import contractions
from nltk.corpus import stopwords
from gensim.models import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from rake_nltk import Rake
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from utils.clean_category import *

tqdm.pandas()
# set the argument parser
parser = argparse.ArgumentParser(
    description="Unsupervised learning on NAICS industry classification."
)

parser.add_argument(
    "--digits",
    type=int,
    default=2,
    help="Number of digits for NAICS codes (2, 4, or 6).",
)

# unpack the arguments
args = parser.parse_args()
# set the arguments
digits = args.digits

MAX_NAICS = 20
MAX_BUSINESS = 20
MAX_TOKENS = 5000
# set path variables
# business_path = "data/data_processed/matched_business.csv"
# web_path = "data/data_processed/matched_web.csv"
review_path = "data/data_processed/llm_syn_review.csv"
# scrapped_path = "data/data_processed/df_merged_carpe_web_test.csv"
embedding_model = "all-MiniLM-L6-v2"

warnings.filterwarnings("ignore")

# Load the embedding model
model = SentenceTransformer(embedding_model)


def embedding(v1, m):
    v1 = v1.lower()
    v1 = v1.replace("-", " ")
    v1 = re.sub('[-,().!?\\"_]', "", v1)
    sep_class_word = v1.split(" ")
    sep_class_word_len = 1
    # NOTE: There can be cases where the word is not in the model
    # In that case, set the vector to zero vector
    vector = np.zeros(m["test"].shape)
    for word in sep_class_word:
        if word == "":
            continue
        try:
            sep_class_word_len += 1
            if word == "&":
                word = "and"
            if word == "/":
                word = "or"
            vector += m[word]
        except KeyError:
            continue
    # Calculate average for the class vector(in this case class is just category)
    vector = vector / sep_class_word_len
    return vector


# class ModelClass:

#     def __init__(self, dim: int = 200):
#         import gensim.downloader as api

#         self.model = api.load("glove-twitter-{}".format(dim))

#     def encode(self, text):
#         return np.array([embedding(word, self.model) for word in text])


# model = ModelClass(dim=200)


stopwords = stopwords.words("english")
# Some extra steps to clean the naics titles
stopword_ext = [
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
na_types = [None, "novalue", "None", np.nan, "nan", "none"]


def to_list(x):
    try:
        return ast.literal_eval(x)
    except:
        return []


def get_nonna_idx(df, col):
    """
    Get the non-null values of a specific column in a DataFrame.
    """

    return (~df[col].isin(na_types)) & (df[col].notna())


# List of irrelevant phrases to remove
irrelevant_phrases = [
    "skip to main content",
    "skip to content",
    "home",
    "privacy policy",
    "terms and conditions",
    "terms of use",
    "sitemap",
    "login",
    "contact us",
    "about us",
    "careers",
    "find a location",
    "read more",
    "learn more",
    "learn more about",
    "view all",
    "all rights reserved",
    "rights reserved",
    "click here to",
    "sign up for",
    "join our team",
    "request a quote",
    "do not sell",
    "my personal information",
    "code of conduct",
    "connect with us",
    "find out more",
    "for more information",
]


# Preprocessing function
def preprocess_text(text):
    # Handle missing or non-string text
    if pd.isna(text):
        return None

    # Remove HTML tags and convert to lowercase
    text = BeautifulSoup(text, "html.parser").get_text().lower()

    # Expand contractions
    text = contractions.fix(text)

    # Remove email addresses, URLs
    text = re.sub(r"\S+@\S+|http\S+|www\S+", "", text)

    # Remove irrelevant phrases
    for phrase in irrelevant_phrases:
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", "", text)

    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Truncation function (same as before)
def truncate_text(text, max_tokens=MAX_TOKENS):
    if pd.isnull(text):
        return ""
    tokens = str(text).split()
    if len(tokens) > max_tokens:
        return " ".join(tokens[:max_tokens])
    else:
        return text


# match companies with business information
def data_processing(review_df):
    """
    Process the review_df to extract title/description nouns for NAICS categorization and business description.
    """

    # clean the reviews
    review_df = data_clean(review_df, "content", "content_clean")
    review_df["content_clean_list"] = review_df["content_clean"].apply(
        lambda x: word_tokenize(x)
    )
    review_df = lemma_noun_df(review_df, "content_clean", "content_clean_noun")

    # drop empty categories
    review_df = review_df[review_df["categories"] != "[]"]
    # clean category data
    review_df = clean_class_df(review_df, "categories")
    review_df = remove_stopwords_df(review_df, "category_clean")
    review_df = lemmatize_class(review_df, "category_clean")
    # select only with categories
    review_df = review_df[review_df["category_clean"].str.len() > 0].reset_index()

    # drop entries with empty cotent_clean/category_clean
    review_df = review_df[review_df["content_clean"].notnull()]
    review_df = review_df[review_df["category_clean"].notnull()]

    return review_df


def recode_naics(naics, digits: int = 2):
    """
    Recode the NAICS codes to 2/4/6 digits.
    """
    if np.isnan(naics) or naics in na_types:
        return naics
    elif digits == 2:
        new_naics = int(naics // 10000)
        if new_naics >= 31 and new_naics <= 33:
            return "31-33"
        elif new_naics >= 44 and new_naics <= 45:
            return "44-45"
        elif new_naics >= 48 and new_naics <= 49:
            return "48-49"
        else:
            return str(new_naics)
    elif digits == 4:
        return str(naics // 100)
    elif digits == 6:
        return str(naics)


# prepare 2/4/6 digits NAICS codes
def prepare_naics(digits: int = 2):
    """
    Prepare the NAICS codes for 2/4/6 digits.
    """

    def _remove_stopwords(text, stopwords=stopwords):
        """
        Remove stopwords from the text.
        """
        return " ".join([word for word in text.split(" ") if word not in stopwords])

    # read in the data
    naics = pd.read_csv(
        "data/naics/6-digits_2022.csv", index_col=False, header=0, sep="\t"
    )
    # rename the columns
    naics.columns = ["naics", "title", "description"]
    # convert the columns to string
    naics["title"] = naics["title"].astype(str).apply(lambda x: " ".join(x.split()))
    # naics["index"] = naics["index"].astype(str).apply(lambda x: " ".join(x.split()))
    naics["description"] = naics["description"].astype(str).apply(preprocess_text)
    # get all NAICS titles & descriptions from 6-digit NAICS to fill 2/4-digit NAICS
    # rename 6-digit NAICS codes to 2/4-digit NAICS codes
    naics["naics"] = naics["naics"].apply(lambda x: recode_naics(int(x), digits=digits))

    # group by new NAICS codes and group all titles and descriptions as lists
    naics = naics.groupby("naics").agg(lambda x: list(x))
    # set index as new column and reset the index
    naics["naics"] = naics.index
    naics = naics.reset_index(drop=True)
    # de-deuplicate the NAICS titles, index and descriptions
    naics["title"] = naics["title"].apply(lambda x: list(set(x)))
    # NOTE: index can be utilized the same way as description if needed
    # naics["index"] = naics["index"].apply(lambda x: " ".join(list(set(x))))
    naics["description"] = naics["description"].apply(lambda x: " ".join(list(set(x))))
    # # split index into a list
    # naics["index"] = naics["index"].apply(
    #     lambda x: list(set(sum([re.split(", |and ", _x) for _x in x], [])))
    # )
    # clean title, index and description
    naics["title"] = naics["title"].apply(lambda x: [clean_desc(_x) for _x in x])
    # naics["index"] = naics["index"].apply(lambda x: clean_desc(x))
    naics["description"] = naics["description"].apply(lambda x: clean_desc(x))

    # remove stopwords in title and index
    stopwords.extend(stopword_ext)
    naics["title"] = naics["title"].apply(
        lambda x: [_remove_stopwords(_x, stopwords=stopwords) for _x in x]
    )
    # naics["index"] = naics["index"].apply(
    #     lambda x: _remove_stopwords(x, stopwords=stopwords)
    # )
    naics["description"] = naics["description"].apply(
        lambda x: _remove_stopwords(x, stopwords=stopwords)
    )

    # clean description
    naics = data_clean(naics, "description", "description_clean")
    naics["description_clean_list"] = naics["description_clean"].apply(
        lambda x: word_tokenize(x)
    )
    naics = lemma_noun_df(naics, "description_clean", "description_clean_noun")

    return naics


# apply LDA
def topic_LDA(
    df,
    noun_col: str = "content_clean_noun",
    list_col: str = "content_clean_list",
    top_topics: int = 10,
    min_count: int = 5,
):
    """
    Apply LDA to the df to extract topics.
    """

    def _apply_lda(text):
        try:
            # Put words into dictionary and bag-of-words to prepare for LDA
            dictionary = Dictionary([text])
            corpus = [dictionary.doc2bow(word) for word in [text]]

            # apply LDA
            lda = LdaMulticore(corpus, num_topics=1)
            # take top topics
            topic = [t[0] for t in lda.get_topic_terms(0, top_topics)]
            return [dictionary[t] for t in topic]
        except Exception as e:
            # if exception exists: return empty list
            return []

    # apply LDA to the noun/list column
    # if noun_col is smaller than min_count, use the list_col
    # apply version
    # return df.apply(
    #     lambda x: (
    #         _apply_lda(x[noun_col])
    #         if len(x[noun_col]) > min_count
    #         else _apply_lda(x[list_col])
    #     ),
    #     axis=1,
    # )
    # for loop version
    results = []
    for i, c in tqdm(df.iterrows(), total=df.shape[0]):
        # check if min_count is met
        if len(c[noun_col]) > min_count:
            # apply LDA to the noun column
            results.append(_apply_lda(c[noun_col]))
        else:
            # apply LDA to the list column
            results.append(_apply_lda(c[list_col]))

    return results


# apply Rake to get topics
def topic_rake(
    df,
    cols: list = ["webpage", "description"],
    top_topics: int = 10,
    max_length: int = 4,
):
    # apply version
    # def _apply_rake(text):
    #     if text != "novalue":
    #         # apply rake to the text
    #         rake = Rake(stopwords=stopwords, max_length=max_length)
    #         rake.extract_keywords_from_text(text)
    #         # get the top topics
    #         return rake.get_ranked_phrases()[:top_topics]
    #     else:
    #         return []

    # # apply rake to the columns
    # return df.apply(
    #     lambda x: sum([_apply_rake(x[col]) for col in cols], []),
    #     axis=1,
    # )
    # for loop version
    rake = Rake(stopwords=stopwords, max_length=max_length)
    results = []

    for i, c in tqdm(df.iterrows(), total=df.shape[0]):
        result = []
        # iterate through the columns
        for col in cols:
            # check if the column is not "novalue"
            if c[col] not in na_types:
                # apply rake to the text
                rake.extract_keywords_from_text(c[col])
                # get the top topics
                result.extend(rake.get_ranked_phrases()[:top_topics])
        # flatten the list and remove duplicates
        result = list(set(result))
        # append the result to the results list
        results.append(result)
    return results


def unsupervised_learning(
    review,
    naics,
    top_naics: int = 6,
    top_business: int = 6,
    top_sim: int = 6,
    top_result: int = 3,
):
    # check top_sim to be smaller than top_naics * top_business
    if top_sim > top_naics * top_business:
        raise ValueError(
            "top_sim should be smaller than top_naics * top_business, please check the parameters."
        )

    # Use helper function to weight entire df
    def cal_weight(category, column="topic_lda", embed_col="category_clean"):
        all_weights = []
        for i, c in category.iterrows():
            if isinstance(c[column], np.ndarray) and len(c[column]) > 0:
                temp = np.mean(cosine_similarity(c[embed_col], c[column]), axis=0)
                # temp = [
                #     np.mean(model.similarity(c[embed_col], topic))
                #     for topic in c[column]
                # ]
                if sum(temp) == 0:
                    all_weights.append([1] * len(c[column]))
                else:
                    all_weights.append(temp)
            else:
                all_weights.append([1] * len(c[column]))
        return all_weights

    def get_top_similar(df, col, order_col, top_k: int = 6):
        """
        Get the top k similar embeddings based on weights.
        """

        results = []
        for i, row in df.iterrows():
            values, order = row[col], np.argsort(row[order_col])
            # if values is smaller than top_k, get all values
            if len(values) < top_k:
                warnings.warn(
                    f"Length of values is smaller than top_k: {len(values)} < {top_k}"
                )
                results.append(values)
            else:
                # get the top k values
                results.append(np.array(values)[order[-top_k:]])

        return results

    # get top similar embeddings to each naics category
    def _get_top_k(x, top_k: int = 6):
        if len(x) < top_k:
            warnings.warn(
                f"Length of similarities is smaller than top_sim: {len(x)} < {top_k}"
            )
            return x
        else:
            return np.array(x)[np.argsort(x)[-top_k:]]

    # calculate weights for all embeddings
    review["topic_weight"] = cal_weight(review, "topic_embedding", "category_embedding")
    naics["topic_weight"] = cal_weight(naics, "topic_embedding", "title_embedding")

    # get top naics/business embeddings
    review["topic_embedding"] = get_top_similar(
        review, "topic_embedding", "topic_weight", top_k=top_business
    )
    naics["topic_embedding"] = get_top_similar(
        naics, "topic_embedding", "topic_weight", top_k=top_naics
    )

    # calculate similarity between embeddings
    review["similarity"] = review["topic_embedding"].apply(
        lambda x: [
            cosine_similarity(x, naics_embedding).flatten()
            for naics_embedding in naics["topic_embedding"]
        ]
    )

    # select top_sim similar scores and average them
    review["similarity"] = review["similarity"].apply(
        lambda x: [np.mean(_get_top_k(_x, top_sim)) for _x in x]
    )

    # return top similar naics categories
    return review["similarity"].apply(
        lambda x: [naics["naics"][i] for i in np.argsort(x)[::-1][:top_result]]
    )


# merge list columns into a single list column
def combine_list_columns(row, cols):
    combined = []
    for col in cols:
        if isinstance(row[col], list):
            combined.extend(row[col])

    return list(set(combined))


def get_accuracy(review_clean, label_cols: list = ["naics1"], verbose: bool = False):

    # top-k accuracy
    # if intersection of the predicted and actual labels is not empty, then the prediction is correct
    def _accuracy(pred, actual):
        return len(set(pred) & set(actual)) > 0

    # calculate the accuracy of the predictions
    # get top-1 prediction accuracy
    preds = review_clean["naics_pred"].apply(lambda x: x[0])
    # merge all label columns into a single list column
    review_clean["naics_recode"] = review_clean[label_cols].values.tolist()
    # remove empty labels & convert each to integer
    review_clean["naics_recode"] = review_clean["naics_recode"].apply(
        lambda x: [int(i) for i in x if str(i) != "nan"]
    )
    # recode the labels to 2/4/6 digits
    review_clean["naics_recode"] = review_clean["naics_recode"].apply(
        lambda x: [recode_naics(_x, digits) for _x in x]
    )
    accuracy_1 = sum(
        [pred in _class for _class, pred in zip(review_clean["naics_recode"], preds)]
    ) / len(review_clean)

    # get top-2 prediction accuracy
    preds = review_clean["naics_pred"].apply(lambda x: x[:2])
    accuracy_2 = sum(
        [
            _accuracy(pred, _class)
            for _class, pred in zip(review_clean["naics_recode"], preds)
        ]
    ) / len(review_clean)

    # get top-3 prediction accuracy
    preds = review_clean["naics_pred"]
    accuracy_3 = sum(
        [
            _accuracy(pred, _class)
            for _class, pred in zip(review_clean["naics_recode"], preds)
        ]
    ) / len(review_clean)

    if verbose:
        print("Top-1 prediction accuracy: {:.2f}%".format(accuracy_1 * 100))
        print("Top-2 prediction accuracy: {:.2f}%".format(accuracy_2 * 100))
        print("Top-3 prediction accuracy: {:.2f}%".format(accuracy_3 * 100))

    return accuracy_1, accuracy_2, accuracy_3


def exp(review_clean, naics, top_naics, top_business, top_sim):
    # get the top similar NAICS categories predictions
    review_clean["naics_pred"] = unsupervised_learning(
        review_clean,
        naics,
        top_naics=top_naics,
        top_business=top_business,
        top_sim=top_sim,
    )
    # get the accuracy of the predictions
    return get_accuracy(
        review_clean,
        label_cols=["naics"],
        verbose=False,
    )


def exp_wrapper(review_clean, naics, args):
    return exp(review_clean, naics, *args)


def plot_result(result, digits: int = 2):
    # get index and values
    naics = [i[0] for i in result.keys()]
    business = [i[1] for i in result.keys()]
    sim = [i[2] for i in result.keys()]
    # get the accuracy values
    top_1 = [i[0] for i in result.values()]
    top_2 = [i[1] for i in result.values()]
    top_3 = [i[2] for i in result.values()]

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.colors import Normalize

    mpl.rcParams["xtick.labelsize"] = 12
    mpl.rcParams["ytick.labelsize"] = 12

    # creating figures
    fig = plt.figure(figsize=(25, 10))
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    # norm = Normalize(vmin=round(min(top_1), 1), vmax=round(max(top_3), 1) + 0.1)
    norm = Normalize(vmin=0, vmax=0.8)

    # top-1 accuracy
    # creating the heatmap
    img1 = ax1.scatter(naics, business, sim, c=top_1, cmap="viridis", s=100, norm=norm)
    # adding title and labels
    ax1.set_xlabel("Top NAICS words", fontsize=16)
    ax1.set_ylabel("Top Business words", fontsize=16)
    ax1.set_zlabel("Top Scoring pairs", fontsize=16)
    ax1.set_title("Top-1 Accuracy of {}-digit Model".format(digits), fontsize=20)
    ax1.set_xticks(np.arange(min(naics), max(naics) + 1, 2))
    ax1.set_yticks(np.arange(min(business), max(business) + 1, 2))
    ax1.invert_xaxis()

    # top-2 accuracy
    # creating the heatmap
    img2 = ax2.scatter(naics, business, sim, c=top_2, cmap="viridis", s=100, norm=norm)
    # adding title and labels
    ax2.set_xlabel("Top NAICS words", fontsize=16)
    ax2.set_ylabel("Top Business words", fontsize=16)
    ax2.set_zlabel("Top Scoring pairs", fontsize=16)
    ax2.set_title("Top-2 Accuracy of {}-digit Model".format(digits), fontsize=20)
    ax2.set_xticks(np.arange(min(naics), max(naics) + 1, 2))
    ax2.set_yticks(np.arange(min(business), max(business) + 1, 2))
    ax2.invert_xaxis()

    # top-3 accuracy
    # creating the heatmap
    img3 = ax3.scatter(naics, business, sim, c=top_3, cmap="viridis", s=100, norm=norm)
    # adding title and labels
    ax3.set_xlabel("Top NAICS words", fontsize=16)
    ax3.set_ylabel("Top Business words", fontsize=16)
    ax3.set_zlabel("Top Scoring pairs", fontsize=16)
    ax3.set_title("Top-3 Accuracy of {}-digit Model".format(digits), fontsize=20)
    ax3.set_xticks(np.arange(min(naics), max(naics) + 1, 2))
    ax3.set_yticks(np.arange(min(business), max(business) + 1, 2))
    ax3.invert_xaxis()

    cbar = fig.colorbar(img1, ax=[ax1, ax2, ax3], shrink=0.6)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Accuracy", fontsize=16)
    # plt.tight_layout()
    plt.savefig(
        "hyperparameter_impact_{}_digit.pdf".format(digits),
        dpi=300,
        bbox_inches="tight",
    )
    # displaying plot
    plt.show()


# loop through hyperparameters to get all combinations
def _get_combination(max_naics, max_business):
    from itertools import chain

    top_naics_list = [i for i in range(2, max_naics + 1, 2)]
    top_business_list = [i for i in range(2, max_business + 1, 2)]
    combinations = [
        [(i, j, k) for k in [top_sim for top_sim in range(2, i * j + 1, 2)]]
        for i in top_naics_list
        for j in top_business_list
    ]

    return list(chain.from_iterable(combinations))


def main(digits: int = 2):

    # check if the preprocessed data files exist
    review_clean_path = "data/data_processed/review_clean.pkl"
    naics_path = "data/data_processed/naics_clean_{}.pkl".format(digits)

    # check if the preprocessed data files exist
    if not os.path.exists(review_clean_path):
        # read in the data
        review_df = pd.read_csv(review_path, index_col=False)

        # process the data
        review_clean = data_processing(review_df)
        # apply LDA to the data
        review_clean["lda_topics"] = topic_LDA(
            review_clean,
            noun_col="content_clean_noun",
            list_col="content_clean_list",
            top_topics=10,
            min_count=5,
        )
        # apply rake to the data
        review_clean["rake_topics"] = topic_rake(
            review_clean,
            cols=["content_clean"],
            top_topics=10,
            max_length=4,
        )
        # merge the lda and rake topics
        review_clean["topics"] = review_clean.apply(
            lambda x: combine_list_columns(
                x, ["category_clean", "lda_topics", "rake_topics"]
            ),
            axis=1,
        )
        # pre-embedding and reading as dataframe? cannot read the vector
        # embed NAICS/business information to numerical vectors
        review_clean["topic_embedding"] = review_clean["topics"].apply(model.encode)
        review_clean["category_embedding"] = review_clean["category_clean"].apply(
            model.encode
        )
        # save the processed data
        review_clean.to_pickle(review_clean_path)
    else:
        # read in the data
        review_clean = pd.read_pickle(review_clean_path)
        # # convert string list columns to list columns
        # list_cols = [
        #     "categories",
        #     "category_clean",
        #     "content_clean_noun",
        #     "content_clean_list",
        #     "lda_topics",
        #     "rake_topics",
        #     "topics",
        #     "topic_embedding",
        #     "category_embedding",
        # ]
        # for col in list_cols:
        #     review_clean[col] = review_clean[col].apply(to_list)

    # check if the NAICS codes exist
    if not os.path.exists(naics_path):
        # prepare the NAICS codes
        naics = prepare_naics(digits=digits)
        # apply LDA to the NAICS codes
        naics["lda_topics"] = topic_LDA(
            naics,
            noun_col="description_clean_noun",
            list_col="description_clean_list",
            top_topics=10,
            min_count=5,
        )
        # apply rake to the NAICS codes
        naics["rake_topics"] = topic_rake(
            naics,
            cols=["description"],
            top_topics=10,
            max_length=4,
        )
        # merge the title and lda/rake topics
        naics["topics"] = naics.apply(
            lambda x: combine_list_columns(x, ["title", "lda_topics", "rake_topics"]),
            # lambda x: combine_list_columns(x, ["title"]),
            axis=1,
        )
        # embed NAICS/business information to numerical vectors
        naics["topic_embedding"] = naics["topics"].apply(model.encode)
        naics["title_embedding"] = naics["title"].apply(model.encode)
        # save the processed data
        naics.to_pickle(naics_path)
    else:
        # read in the data
        naics = pd.read_pickle(naics_path)
        # # convert string list columns to list columns
        # list_cols = [
        #     "title",
        #     "description_clean_list",
        #     "description_clean_noun",
        #     "lda_topics",
        #     "rake_topics",
        #     "topics",
        #     "topic_embedding",
        #     "title_embedding",
        # ]
        # for col in list_cols:
        #     naics[col] = naics[col].apply(to_list)

    combinations = _get_combination(MAX_NAICS, MAX_BUSINESS)
    # parallelize the unsupervised learning using multiprocessing
    from multiprocessing import Pool
    from functools import partial

    target = partial(exp_wrapper, review_clean, naics)
    with Pool(processes=18) as pool:
        result = list(tqdm(pool.imap(target, combinations), total=len(combinations)))

    # convert the result to a DataFrame
    pd.DataFrame({"combinations": combinations, "accuracy": result}).to_csv(
        "result_{}.csv".format(digits), index=False
    )
    # concat results as dict
    result = dict(zip(combinations, result))
    # plot the results
    plot_result(result, digits=digits)

    # get the best combination
    best_combination = max(result, key=lambda x: result[x][0])
    print(
        "Best combination:\nTop naics words:{}, Top business words: {}, Top scoring pairs: {}.\n Accuracy: {}.".format(
            best_combination[0],
            best_combination[1],
            best_combination[2],
            result[best_combination],
        )
    )


if __name__ == "__main__":
    main(digits=digits)
