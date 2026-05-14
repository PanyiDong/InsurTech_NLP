# -*- coding: utf-8 -*-
"""Extracting, processing and storing row risk data.
"""


import json
import csv
import pandas as pd
from os import listdir
from os.path import isfile, join
import re
from tqdm import tqdm


def read_file(path: str, companyList: list):
    """
    This function read a json file and spit out companyList for each business in the file
    *This function takes one file at a time*
    :param path: Path of the file that one wants to read
    :param companyList: a list of company data to be appended
    :return: return a list of json object that contains information for each business
    """
    print("reading data from {}...".format(path))
    try:
        with open(path, encoding="utf-8") as f:
            for jsonObj in f:
                companyDict = json.loads(jsonObj)
                if companyDict["id"] == "00446c7f58fdd906":
                    print(companyDict)
                companyList.append(companyDict)
        f.close()
    except UnicodeDecodeError:
        with open(path, errors="ignore") as f:
            for jsonObj in f:
                companyDict = json.loads(jsonObj)
                companyList.append(companyDict)
        f.close()
    print("read {} items.".format(len(companyList)))
    return companyList


def write_reviews(path: str, companyList: list):
    """
    This function write the 'reviews' json sub object from the main json object
    :param path: The path of the file that one wants to write at
    :param companyList: a list of json object that contains information for each business,
    we retrieve 'reviews' sub object specifically for this function
    :return: Nothing to be returned, print done when successfully wrote the file
    """

    print("writing reviews...")
    with open(path, mode="w", newline="\n", encoding="utf-8") as csv_file1:
        # get the reviews object
        second_lvl = []
        for c in companyList:
            if c["reviews"] is not None:
                second_lvl = list(c["reviews"][0].keys())
                break
        header = ["company_duns", "company_id"] + second_lvl
        # initiate writer and write header
        writer = csv.DictWriter(csv_file1, fieldnames=header)
        writer.writeheader()

        # The following loop loop through each business' information in the companyList
        for i in tqdm(range(len(companyList))):
            # if there is no review, write none for all columns
            if companyList[i]["reviews"] is None:
                csv_file1.write(companyList[i]["company_duns"] + ",")
                csv_file1.write(companyList[i]["id"] + ",")

                for k in second_lvl:
                    csv_file1.write("novalue" + ",")
                csv_file1.write("\n")
            else:
                # if there are reviews loop through each review in the reviews of that business
                for r in companyList[i]["reviews"]:
                    # write id of the business
                    csv_file1.write(companyList[i]["company_duns"] + ",")
                    csv_file1.write(companyList[i]["id"] + ",")
                    # for each review of that business, loop through the sub-json keys
                    for k in second_lvl:
                        # Some data Cleaning to format the file
                        x = str(r[k]).replace(",", ";")
                        x = x.replace("\n", "")
                        x = x.replace("\r", "")
                        x = re.sub(r"(\n\s*)+", "", x)
                        x = re.sub(r"(\r\s*)+", "", x)
                        # Deal with empty exceptions
                        if str(r[k]) == "":
                            x = "novalue"
                        csv_file1.write(x + ",")
                    # newline for each review.
                    csv_file1.write("\n")


def write_business_info(path: str, companyList: list):
    """
    This function writes the business information from each business's json object
    to a csv file. The selection of features is based on the documentation from carpe.
    Basically, anything that is not risk.indicators are included in this csv file
    :param path: path of the file that you want to write to
    :param companyList: a list of json object that contains information for each business
    :return: return nothing, print done when successfully wrote the file
    """

    print("writing business info...")
    with open(path, mode="w", newline="\n", encoding="utf-8") as csv_file1:
        # This is the first level json keys that this function is going to use,
        # THIS IS NOT ALL THE KEYS
        first_lvl_json_keys = [
            "company_duns",
            "id",
            "address",
            "business",
            "classification",
            "urls",
            "firmo",
            "contacts",
            "hours",
        ]
        # Initiate second_lvl_json_keys as a list, this list is going to contain the keys of the sub-json objects
        second_lvl_json_keys = []

        # This for loop is used to retrieve the second level json keys,
        # one exception is 'classification', since we are only retrieving the 'classification_list' object
        for key in first_lvl_json_keys:
            # Dealing with a few exceptions
            if key == "business":
                second_lvl_json_keys.append(["name", "names"])
                continue

            elif key == "classification":
                second_lvl_json_keys.append(["categories"])
                continue
            # Due to dataset errors, there are json objects that does not have all the keys
            # For dealing with those errors, as well as Nonetypes of dictionary,
            # The following loop was used
            for c in companyList:
                # Using try except to catch KeyError for objects that does not contain all the keys
                # if object has the key, check if it is None, dictionary, or other
                try:
                    if c[key] is not None:
                        if type(c[key]) is dict:
                            # Appending some keys if it is dictionary
                            second_lvl_json_keys.append(list(c[key].keys()))
                            break
                        else:
                            # Appending first_level key if it is not dictionary
                            second_lvl_json_keys.append([key])
                            break
                    else:
                        # continue if None
                        continue
                except "KeyError":
                    continue

        # The following line convert 2d second_lvl_json_keys into one long 1d list
        flat_key = [key for sublist in second_lvl_json_keys for key in sublist]
        flat_key[1] = "company_id"
        # write header of the csv file
        # I chose '`' as my delimiter because it is rarely used.
        # This character will not cause conflict with the commas in the json files
        writer = csv.DictWriter(csv_file1, fieldnames=flat_key, delimiter="\t")
        writer.writeheader()

        # Loop through each object in the companyList
        for i in tqdm(range(len(companyList))):
            for j in range(len(first_lvl_json_keys)):
                # Retrieving first level keys
                key1 = first_lvl_json_keys[j]

                # check if there is a dictionary under this key, if not, write the value
                try:
                    if type(companyList[i][key1]) is not dict:
                        if companyList[i][key1] is None:
                            for m in range(len(second_lvl_json_keys[j])):
                                csv_file1.write("novalue" + "\t")
                            continue
                        else:
                            csv_file1.write(companyList[i][key1] + "\t")
                except "KeyError":
                    if len(second_lvl_json_keys[j]) == 1:
                        csv_file1.write(companyList[i][key1] + "\t")
                        continue
                    else:
                        for n in range(len(second_lvl_json_keys[j])):
                            csv_file1.write("novalue" + "\t")
                        continue

                # if there is a dictionary, write the values within the dictionary by keys
                if type(companyList[i][key1]) is dict:
                    # loop through the second_lvl_json_keys of this particular first level key
                    for k in range(len(second_lvl_json_keys[j])):

                        # get key2, which is a second_lvl_json_keys
                        key2 = second_lvl_json_keys[j][k]
                        x = None
                        if key2 == "categories":
                            x = []
                            for cat in companyList[i][key1][key2]:
                                if cat["origin"] == "naics":
                                    continue
                                print(cat["origin"])
                                x.append(cat["category"])
                        # get the values under this dictionary, cast it as string
                        else:
                            x = companyList[i][key1][key2]
                        x = str(x)
                        # some slight data cleaning
                        x = x.replace("\n", ";")
                        x = x.replace("\t", ";")
                        if x == "":
                            x = "novalue"

                        # write into the file
                        csv_file1.write(x + "\t")
            # write a new line for each business
            csv_file1.write("\n")

    # Since I chose '`' to be my delimiter, the csv file is hard to read,
    # thus the following process of reading it into pandas and then write out a csv is needed
    df = pd.read_csv(filepath_or_buffer=path, sep="\t", header=0, index_col=False)
    df.to_csv(path_or_buf=path, index=False)
    print("Done")


def write_webbusi(path: str, companyList: list, verbose=True, opts=None):
    """
    This function write the 'webpage:content' json sub object from the main json object
    :param path: The path of the file that one wants to write at
    :param companyList: a list of json object that contains information for each business,
    we retrieve 'webpage:content' sub object specifically for this function
    :return: Nothing to be returned, print done when successfully wrote the file
    """

    k_id = ["company_duns", "id"]

    if verbose:
        print("writing webpages...")
    with open(path, mode="w", newline="\n", encoding="utf-8") as csv_file1:
        # get the reviews object

        # initiate writer and write header
        writer = csv.DictWriter(
            csv_file1, fieldnames=k_id + ["webpage", "description", "naics"]
        )
        # writer = csv.DictWriter(csv_file1, fieldnames=['company_id','webpage','description'])
        writer.writeheader()

        # The following loop loop through each webpage information in the companyList
        for i in tqdm(range(len(companyList)), disable=not verbose):
            # if there is no webpage, write NAN for all columns
            web = "novalue"
            busi = "novalue"
            naics = "novalue"

            if companyList[i]["webpage"] is not None:
                web = companyList[i]["webpage"]["content"]

                if web is None:
                    web = "novalue"
                else:
                    # web = (web[:MAX_LENGTH] + '..') if len(web) > MAX_LENGTH else web
                    web = web.replace("\n", "")
                    web = web.replace("\r", "")
                    web = web.replace('"', "")
                    web = re.sub(r"(\n\s*)+", "", web)
                    web = re.sub(r"(\r\s*)+", "", web)

            if companyList[i]["business"] is not None:
                busi = companyList[i]["business"]["description"]
                if busi is None:
                    busi = "novalue"
                else:
                    # busi = (busi[:MAX_LENGTH] + '..') if len(busi) > MAX_LENGTH else busi
                    busi = busi.replace("\n", "")
                    busi = busi.replace("\r", "")
                    busi = busi.replace('"', "")
                    busi = re.sub(r"(\n\s*)+", "", busi)
                    busi = re.sub(r"(\r\s*)+", "", busi)

            if companyList[i]["classification"] is not None:
                naics = companyList[i]["classification"]["naics"]
                if naics is None or len(naics) == 0:
                    naics = "novalue"
                else:
                    naics = naics[0]["code"]
            if naics is None:
                naics = "novalue"

            for ids in k_id:
                csv_file1.write(str(companyList[i][ids]) + ",")
            csv_file1.write(f'"{web}",')
            csv_file1.write(f'"{busi}",')
            csv_file1.write(naics + ",")
            csv_file1.write("\n")
    if verbose:
        print("finished writing webpages.")


def find_data(path: str):
    """
    find all files in the 'Data directory'
    :param path: path of the directory
    :return: return a list of files' name for converting entire folder of data into csv
    """
    # append each file name into files list
    files = [f for f in listdir(path) if isfile(join(path, f))]
    # pop the first element,
    # At least for my version, the list will always start with '.DS_Store' and is basically uselses for us
    # files.pop(0)
    return files


def extract_data(source, target):
    """
    # This is the main function for extracting, processing
    # and storing data.

    # Process data are stored in 4 csv files respectively:
        a. review.csv: contains review organized by business.
        b. index.csv:  contains Carpe index by business.
        c. business.csv: contains business infos.
        d. risk.csv: contains risk indicators by business

    # The four files are matched by attribute company_id.

    # additional_risks.txt will also be produced for a folder of files
    # recording the risk indicators that don't match with the documentation

    # The code continues to run even if an error is thrown out
    # the during the process, and error message will be printed
    # out in those cases.

    # So please check the print log to ensure all the data are
    # vaild.

    :param source: dir contains raw unzipped data (folder)
    :param target: dir where to write csv (folder)
    :return: None
    """
    count = 0
    data_list = find_data(path=source)
    print("{} raw files to extract.".format(len(data_list)))
    companyList = []
    for file in data_list:
        if "waja" in file:
            print("working on {}...".format(file))
            companyList = read_file(join(source, file), companyList)
    try:
        print(len(companyList))
        write_reviews(
            path=join(target, "matched" + "_review.csv"), companyList=companyList
        )
        write_business_info(
            path=join(target, "matched" + "_business.csv"), companyList=companyList
        )
        write_webbusi(
            path=join(target, "matched" + "_web.csv"), companyList=companyList
        )
        count = len(companyList)
    except Exception as e:
        print("An error occurs at {}".format(file))
        print(e)
        # continue
    print("Finished processing data for {} companies".format(count))


if __name__ == "__main__":
    extract_data(source="data/data_raw/", target="data/data_processed/")
