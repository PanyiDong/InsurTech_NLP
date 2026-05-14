import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from textblob import TextBlob
import gensim.downloader as api
import regex as re

m = api.load("glove-twitter-200")


def cos_similarity(v1, v2):
    top = np.dot(v1, v2)
    bottom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if bottom == 0.0:
        return 0.0
    return top / bottom


def word_similarity(v1: str, v2: str):
    # clean the words a little bit
    v1 = v1.lower()
    v1 = v1.replace("-", " ")
    v1 = re.sub('[-,().!?\\"_]', "", v1)

    v2 = v2.lower()
    v2 = v2.replace("-", " ")
    v2 = re.sub('[-,().!?\\"_]', "", v2)

    class_vector = embedding(v1)
    topic_vector = embedding(v2)

    return cos_similarity(class_vector, topic_vector)


def embedding(v1):
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


# Embed a whole list of categories and give an average vector
def embed_list(l1):
    vector_list = []
    if len(l1) == 0:
        return 0
    for i in l1:
        vector_list.append(embedding(i))
    return np.average(vector_list, axis=0)


# Use embed_list function to embed entire dataframe given the dataframe and column
def embed_all(df, embed_col):
    return df[embed_col].apply(lambda x: embed_list(x))


# Weigting categories and apply weighting to entire business df
# There is no weighting on naics files
# A helper function to weight each category with topics
def weight_category(category, review_topics):
    weights = []
    if type(category) == str:
        return [0]
    for cat in category:
        temp = []
        for t in review_topics:
            temp.append(word_similarity(cat, t))
        temp = sorted(temp, reverse=True)[0:5]
        weights.append(np.average(temp))
    return weights


# Use helper function to weight entire df
def weight_all(category, column="topic_lda", embed_col="category_clean"):
    all_weights = []
    for i, c in category.iterrows():
        if c[column] != [] or c[column] != "novalue":
            temp = weight_category(c[embed_col], c[column])
            if sum(temp) == 0:
                all_weights.append([1] * len(c[embed_col]))
            else:
                all_weights.append(temp)
        else:
            all_weights.append([1] * len(c[embed_col]))
    return all_weights


def clean_class(class_str: str):
    # Replacing some unwanted characters in the category columns
    class_str = class_str.lower()
    class_str = class_str.replace("[", "")
    class_str = class_str.replace("]", "")
    class_str = class_str[1:-1]
    class_str = class_str.replace("'s ", " ")
    class_str = class_str.replace('"', ",")
    class_str = class_str.replace(", ", ",")
    class_str_list = re.split("','", class_str)
    # dedup
    return list(set(class_str_list))


def clean_class_df(df: object, col: str):
    # apply previous function to a dataframe
    class_clean = df[col].apply(lambda x: clean_class(x))
    df["category_clean"] = class_clean
    return df


def round_top_n_cat(
    category,
    category_weight_column,
    topic_weight_column,
    topic_column,
    category_column="category_clean",
    n=7,
):
    top_n_cat = []
    for i, c in category.iterrows():
        temp = []
        if len(c[category_column]) < n:
            temp.extend(c[category_column])
            if len(c[topic_weight_column]) != 0:
                top = np.argsort(c[topic_weight_column])[
                    -(n - len(c[category_column])) :
                ]
                top_ndarray = np.array(c[topic_column])
                temp.extend(top_ndarray[top])
            elif len(topic_weight_column) == 0:
                top_n_cat.append(temp)
                continue
        if len(c[category_column]) > n:
            top = np.argsort(c[category_weight_column])[-n:]
            cat_ndarray = np.array(c[category_column])
            temp.extend(cat_ndarray[top])

        if len(c[category_column]) == n:
            temp.extend(c[category_column])
        top_n_cat.append(temp)
    return top_n_cat


def remove_stopwords_title(df: object, col: str, stopwords=stopwords.words("english")):
    # stopwords usually do not mean anything and need to be removed,
    # this fuction removes all the stop words in the review content
    stop = stopwords
    df[col] = df[col].apply(
        lambda x: " ".join([item for item in x.split(" ") if item not in stop])
    )
    return df


# Rank similarities by naics by average simillarity
def rank_similarity(average, rank_type="class"):
    if rank_type == "naics":
        return sorted(average, key=lambda x: x[2], reverse=True)


def classify_embed_per_naics(
    naics, category, index, naics_col, cat_col="embed_clean", top_n_pairs=5
):
    """
    naics: naics dataframe, input can be 2-digit, 4-digit, 6-digit naics code dataframe
    category: category dataframe
    index: Its for specifying a business (This is used for example, in the main classification, every index will be inputed)
    naics_col: Use topics in text, function will embed each topic itself.
    cat_col: using embedding column
    """
    naics_similarity = []
    naics = naics.set_index("Code")

    for index_d, d in naics.iterrows():
        temp = []
        description_topic = d[naics_col]
        temp2 = []
        for dt in description_topic:
            temp.append(cos_similarity(embedding(dt), category.iloc[index][cat_col]))
        temp = np.sort(temp)[-top_n_pairs:]
        naics_similarity.append([index_d, d["Title"], np.average(temp)])

    return np.array(rank_similarity(naics_similarity, "naics")[0:3])


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
    top_n_pairs=6,
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


def calculate_accuracy(category, classification_column, top_n, specify_index=[None]):
    count = 0
    if specify_index[0] != None:
        category = category.iloc[specify_index]
    for i, c in category.iterrows():
        classed_result = c[classification_column][0:top_n]
        for classed_naics in classed_result:
            if "-" in str(classed_naics):
                classed_naics = classed_naics.split("-")
                if (
                    str(c["2_dig_scrapped"]) >= classed_naics[0]
                    and str(c["2_dig_scrapped"]) <= classed_naics[1]
                ):
                    count += 1
                    break
                elif (
                    str(c["2_dig_scrapped_2"]) >= classed_naics[0]
                    and str(c["2_dig_scrapped_2"]) <= classed_naics[1]
                ):
                    count += 1
                    break
            elif str(classed_naics) == str(c["2_dig_scrapped"]) or str(
                classed_naics
            ) == str(c["2_dig_scrapped_2"]):
                count += 1
                break

    print(
        "Accuracy of classification with {classification_column} method selecting top {n} has accuracy {accuracy}.".format(
            classification_column=classification_column,
            n=top_n,
            accuracy=round(count / category.shape[0], 3),
        )
    )

    return round(count / category.shape[0], 3)


def remove_stopwords_class(class_str_list, stopwords=stopwords.words("english")):
    stopwords.extend(
        [
            "office",
            "corporate",
            "service",
            "services",
            "company",
            "companies",
            "business",
            "related",
            "building",
            "place",
            "interest",
            "&",
            "/",
        ]
    )
    clean_list = []
    for l in class_str_list:
        temp_string = " ".join([item for item in l.split(" ") if item not in stopwords])
        if temp_string == "":
            continue
        clean_list.append(temp_string)

    return clean_list


def remove_stopwords_df(df: object, col: str):
    # apply previous function to a dataframe
    class_clean = df[col].apply(lambda x: remove_stopwords_class(x))
    df["category_clean"] = class_clean
    return df


def get_wordnet_pos(word):
    """
    This function is position tagging for nltk lemmatization only
    Map POS tag to first character lemmatize() accepts
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_class(df: object, col: str):
    # Lemmatize class
    # I later found out this doesn't really do much for the categories because it is pretty much in its orignal form already
    complete_lem = []
    for i, c in df.iterrows():
        category = c[col]
        lem_text = []
        for cat in category:
            cat_list = cat.split(" ")
            temp = []
            for word in cat_list:
                temp.append(WordNetLemmatizer().lemmatize(word, get_wordnet_pos(word)))
            lem_text.append(" ".join(temp))
        complete_lem.append(list(set(lem_text)))
    df[col] = complete_lem
    return df


# Cleaning descriptions
def clean_desc(desc_str: str):
    desc_str = desc_str.lower()
    desc_str = desc_str.replace(",", "")
    desc_str = desc_str.replace("(", "")
    desc_str = desc_str.replace(")", "")
    desc_str = desc_str.replace("-", " ")
    if desc_str[0:3] == "non":
        desc_str = "non " + desc_str[3 : len(desc_str)]
    return desc_str


def clean_desc_df(df: object, col_name="Description"):
    df[col_name] = df[col_name].apply(lambda x: clean_desc(x))
    df = data_clean(df, col_name, col_name)
    df[col_name] = tokenize_stem_lemma(
        df, col_name, "desc_token", "desc_stem", "desc_lemma"
    )["desc_lemma"]
    df[col_name] = df[col_name].apply(lambda x: " ".join(x))
    return df


# functions to get nouns only from the naics descriptions
def nouns(text):
    """apply on cleaned text
    return a list of extracted nouns"""
    from nltk import word_tokenize, pos_tag

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
    df[col_final] = df[col_org].apply(lambda x: lemma_noun(nouns(x)))
    return df


def remove_few_content(df, cutoff: int):
    # give a cutoff value and filter out the compaies that have reviews fewer than this cutoff
    # this is to make sure each company have enough reviews to be analyzed
    df = (
        df.groupby("company_id")
        .filter(lambda x: len(x) > cutoff)
        .reset_index(drop=True)
    )
    return df


def contractions_dict():
    # returns a dictionary that have all the common contraction words in English
    contractions_dict = {
        "ain't": "are not",
        "'s": " is",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "that'd": "that would",
        "that'd've": "that would have",
        "there'd": "there would",
        "there'd've": "there would have",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what've": "what have",
        "when've": "when have",
        "where'd": "where did",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who've": "who have",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
    }
    return contractions_dict


def expand_contractions(text, contractions_dict=contractions_dict()):
    # set regular expressions that could be used in the next fuction expand_contractions_col()
    contractions_re = re.compile("(%s)" % "|".join(contractions_dict.keys()))

    def replace(match):
        return contractions_dict[match.group(0)]

    return contractions_re.sub(replace, text)


def expand_contractions_col(df: object, col: str):
    # expand all the contractions words in the text to aviod uncessary new words
    expanded = df[col].apply(lambda x: expand_contractions(x))
    return pd.DataFrame(expanded)


def lower(df: object, col: str):
    # lower all the upper cases words in the text to ensure text consistency
    lowered = df[col].apply(lambda x: str(x).lower())
    return pd.DataFrame(lowered)


def remove_digits(df: object, col: str):
    # remove all the digits in the text as the digits do not contribute in the analysis as the words are and need to be removed
    remove_dig = df[col].apply(lambda x: re.sub("\w*\d\w*", "", x))
    return pd.DataFrame(remove_dig)


def remove_punctuations(df: object, col: str):
    # remove all the puctuations and extra white spaces
    # extra white space means some type in error which uses multiple white spaces to seperate words
    remove_punct1 = pd.DataFrame(
        df[col].apply(lambda x: re.sub("[%s]" % re.escape(string.punctuation), "", x))
    )
    remove_punct2 = remove_punct1[col].apply(lambda x: re.sub(" +", " ", x))
    return pd.DataFrame(remove_punct2)


def remove_stopwords(df: object, col: str):
    # stopwords usually do not mean anything and need to be removed,
    # this fuction removes all the stop words in the review content
    stop = stopwords.words("english")
    stop_removed = df[col].apply(
        lambda x: " ".join([item for item in x.split() if item not in stop])
    )
    return pd.DataFrame(stop_removed)


def data_clean(df: object, col_original: str, col_new: str, rake=False):
    """
    This function combines all the functions above together and do the cleaning job all together
        we separate the orignal text data from the dataframe to do the cleaning for this particular column
        without changing the original data on the dataframe.
    :param df: the dataframe that contains text data
    :param col_original: the column name that contains the text data in that dataframe given in df
    :return: return a dataframe with an added column of cleaned text by name of your choice
    """
    original = pd.DataFrame(df[col_original])
    lowered = lower(original, col_original)
    if rake == False:
        lowered = expand_contractions_col(lowered, col_original)
    remove_dig = remove_digits(lowered, col_original)
    if rake == False:
        remove_punct = remove_punctuations(remove_dig, col_original)
        result = remove_stopwords(remove_punct, col_original)
    else:
        result = remove_stopwords(remove_dig, col_original)
    df[col_new] = result
    return df


def tokenize(df: object, col: str):
    # this function tokenize each review into a list of word
    tokenize_text = df[col].apply(lambda x: word_tokenize(x))
    return pd.DataFrame(tokenize_text)


def stem(df: object, col: str):
    # this function does stemming on the tokenized list
    stem_text = df[col].apply(lambda x: [PorterStemmer().stem(i) for i in x])
    return pd.DataFrame(stem_text)


def textblob_postag_lemmatize(sentence):
    """
    This function is for position tagging for textblob lemmatization only
    Does position taging before lemmatizing
    Used in the general lemmatize function
    """
    sent = TextBlob(sentence)
    tag_dict = {"J": "a", "N": "n", "V": "v", "R": "r"}
    words_and_tags = [(w, tag_dict.get(pos[0], "n")) for w, pos in sent.tags]
    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]
    return lemmatized_list


def lemmatize(df: object, col: str, lemmatizer="nltk"):
    # this function does lemmatization on the tokenized list
    if lemmatizer.lower() == "nltk":
        lem_text = df[col].apply(
            lambda x: [WordNetLemmatizer().lemmatize(i, get_wordnet_pos(i)) for i in x]
        )
    if lemmatizer.lower() == "textblob":
        lem_text = df[col].apply(lambda x: textblob_postag_lemmatize(" ".join(x)))
        lem_text = df[col].apply(lambda x: [i.lemma_ for i in parse(" ".join(x))])
    return pd.DataFrame(lem_text)


def tokenize_stem_lemma(
    df: object,
    col_original: str,
    col_tokenize: str,
    col_stem: str,
    col_lemma: str,
    lemmatizer="nltk",
):
    """
    This function combines tokenization, stemming, and lemmatization into one general function,
    :param df: the dataframe that contains some text, preferably cleaned text
    :param col_original: the column name that contains the text data in that dataframe given in df
    :param col_tokenize: the column name that you want to put your tokenized text under
    :param col_stem: the column name that you want to put your stemmed text under
    :param col_lemma: the column name that you want to put your lemmatized text under
    """
    original = pd.DataFrame(df[col_original])
    tokenized = tokenize(original, col_original).rename(
        columns={col_original: col_tokenize}
    )
    stemmed = stem(tokenized, col_tokenize).rename(columns={col_tokenize: col_stem})
    lemmatized = lemmatize(tokenized, col_tokenize, lemmatizer=lemmatizer).rename(
        columns={col_tokenize: col_lemma}
    )
    new_df = pd.concat([df, tokenized, stemmed, lemmatized], axis=1)
    return new_df


def remove_no_english(df: object, col: str):
    # remove words which doesn't belong to English language
    words = set([x.lower() for x in nltk.corpus.words.words()])
    df["noeng"] = df[col].apply(
        lambda x: [item for item in x if item not in words or not item.isalpha()]
    )
    df[col] = df[col].apply(
        lambda x: [item for item in x if item in words and item.isalpha()]
    )
    return df
