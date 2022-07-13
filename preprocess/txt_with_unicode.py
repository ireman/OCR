from glob import glob
import os
import pandas as pd
import csv
REPLACEMENT_MAP = {
    "š": "sz",
    "ṣ": "s,",
    "ṭ": "t,",
    "ĝ": "j",
    "ḫ": "h",
    # Replace 'smart' quotes with normal characters
    "‘": "'",
    "’": "'",
    "ʾ": "'",
    "“": '"',
    "”": '"',
    # Replace em-dash and en-dash with normal dash
    "–": "-",
    "—": "-",
}
ACUTE_VOWELS = {"á": "a", "é": "e", "í": "i", "ú": "u"}
GRAVE_VOWELS = {"à": "a", "è": "e", "ì": "i", "ù": "u"}

# Extend the dictionaries at import time to include uppercase versions
REPLACEMENT_MAP.update(
    {key.upper(): value.upper() for key, value in REPLACEMENT_MAP.items()}
)
ACUTE_VOWELS.update({key.upper(): value.upper() for key, value in ACUTE_VOWELS.items()})
GRAVE_VOWELS.update({key.upper(): value.upper() for key, value in GRAVE_VOWELS.items()})
'''IMAGE_PATH = "labels_names"
labels = os.listdir(IMAGE_PATH)
name_list =[]
labels_list = []
counter = 0
label_unicode = {}
key = []
unicode = []'''
with open('signs.txt', 'r', encoding='utf-8') as infile:
    list_of_lines = [(line.strip()).split() for line in infile]
    SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉","0123456789")
    res = [''.join(ele[0]).translate(SUB).split('(')[0] for ele in list_of_lines]
    with open('texts_13_09.txt', 'r',encoding='utf-8') as file:
      filedata = file.read()
    word_list = []
    # Replace the target string
    for label in filedata.split():
        for original, replacement in REPLACEMENT_MAP.items():
            label = label.replace(original, replacement)

            # Add the number 2 to the token for acute vowels
        for original, replacement in ACUTE_VOWELS.items():
            if original in label:
                label += "2"
            label = label.replace(original, replacement)

            # Add the number 3 to the token for grave vowels
        for original, replacement in GRAVE_VOWELS.items():
            if original in label:
                label += "3"
            label = label.replace(original, replacement)

        if label not in word_list:
            word_list.append(label)
            try:
                pos = res.index(label)
                filedata = filedata.replace(' '+label+' ', ' '+list_of_lines[pos][2]+' ')
            except ValueError:
                pass
    # Write the file out again
    with open('texts_13_09_unicode1.txt', 'w',encoding='utf-8') as file:
      file.write(filedata)
