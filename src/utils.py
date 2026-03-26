"""
Utilities module
"""


import os
import shutil

import kagglehub


os.environ["KAGGLEHUB_CACHE"] = "data"

def download_kaggle_dataset():

    kagglehub.dataset_download("sushovansaha9/flickr-logos-27-dataset")

    shutil.move(
        "data/datasets/sushovansaha9/flickr-logos-27-dataset/versions/1/flickr_logos_27_dataset",
        "data/"
    )

    shutil.rmtree("data/datasets")
    os.remove("data/flickr_logos_27_dataset/flickr_logos_27_dataset_distractor_set_urls.txt")

    with open("data/flickr_logos_27_dataset/flickr_logos_27_dataset_query_set_annotation.txt", "r", encoding="utf-8") as f:
        content = f.read()

    # Replace tabs with a single space
    content = content.replace("\t", " ")

    with open("data/flickr_logos_27_dataset/flickr_logos_27_dataset_query_set_annotation.txt", "w", encoding="utf-8") as f:
        f.write(content)