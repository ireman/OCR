from glob import glob
import os
import pandas as pd
import csv
import copyfile
from collections import Counter
REPLACEMENT_MAP = {
    "š": "sz",
    "ṣ": "s,",
    "ṭ": "t,",
    "ĝ": "j",
    "ḫ": "h",
    # Subscripted numbers correspond to actual numbers in the original
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
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

IMAGE_PATH = r"C:\Users\Home\Downloads\consensus_data11"
labels = os.listdir(IMAGE_PATH)
name_list =[]
labels_list = []
df = pd.DataFrame()
'''df['label'] = labels
df.to_csv(path_or_buf='consensus_data11_labels.csv', index=None, header=True, encoding='utf-8')
exit()'''
'''for l in labels:
    for image_file in glob("{0}\{1}\*.png".format(IMAGE_PATH, l)):
        fname = os.path.basename(image_file)
        label_name = os.path.dirname(image_file).split('\\')[1]
        pos = fname.replace('_', 'i', 4).find('_')
        fname = fname[pos+1:]
        name_list.append(fname)
        labels_list.append(label_name)
        dname = os.path.dirname(image_file)
        os.rename(image_file, "consensus_data\{}".format(fname))
data = {'image': name_list, 'label': labels_list}
df = pd.DataFrame(data)
df.to_csv(path_or_buf='img_name_and_labels3.csv', index=None, header=True)
for l in labels:
    for image_file in glob("{0}\{1}\*.png".format(IMAGE_PATH, l)):
        fname = os.path.basename(image_file)
        pos = fname.replace('_', 'i', 4).find('_')
        fname = fname[pos:]
        dname = os.path.dirname(image_file)
        os.rename(image_file, "{}{}".format(dname, fname))
for char in characters:
    prefix = char + "_"
    [os.rename(f, "{}{}".format(f, prefix)) for f in glob("{0}/{1}/*.png".format(IMAGE_PATH, char))]'''

i =0
df = pd.read_csv('consensus_data\labels_1+.csv', skipinitialspace=True)

with open('consensus_data\labels_1+.csv', mode='r', newline='') as csv_file:
    for image_file in glob("{0}\*\*".format(IMAGE_PATH)):
        fname = os.path.basename(image_file)
        label = image_file.split('\\')[-2]
        name_list.append(fname)
        labels_list.append(label)
        #copyfile.copyFile(image_file, r'C:\Users\Home\PycharmProjects\OCR\consensus_data11_all')

        '''employee_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if i >= 1023:
            a = 5
        employee_writer.writerow([fname, label])
        i += 1'''
df = pd.DataFrame()
df['image'] = name_list
df['label'] = labels_list
a = Counter(name_list).most_common()
a.reverse()
df.to_csv(path_or_buf='img_labels_unicode_data11.csv', index=None, header=True, encoding='utf-8')

exit()
counter = 0
label_unicode = {}
label_unicode1 = {}

key = []
unicode = []
with open('signs.txt', 'r', encoding='utf-8') as infile:
    list_of_lines = [(line.strip()).split() for line in infile]
    SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉","0123456789")
    res = [''.join(ele[0]).translate(SUB).split('(')[0] for ele in list_of_lines]
    label_unicode_list = []

    label_list = list(df['label'])
    label_list1 = []
    for label in label_list:
        try:
            pos = res.index(label)
            label_unicode[list_of_lines[pos][2]] = label
            label_unicode1[label] = list_of_lines[pos][2]

            label_unicode_list.append(list_of_lines[pos][2])
            counter += 1
            #print(label)
            #print(list_of_lines[pos][2].encode("unicode_escape"))
            key.append(label)
            unicode.append(list_of_lines[pos][2])
        except ValueError:
            label_list1.append(label)
    list_key = list(label_unicode.keys())
    name_list = label_list1+list_key

    data = {'label': name_list}
    df = pd.DataFrame(data)
    df.to_csv(path_or_buf='consensus_data\labels_with_unicode.csv', index=None, header=True, encoding='utf-8')
    data = {'label': key, 'unicode': unicode}
    df = pd.DataFrame(data)
    df.to_csv(path_or_buf='consensus_data\labels_unicode.csv', index=None, header=True, encoding='utf-8')
    df = pd.read_csv('img_name_and_labels3.csv', skipinitialspace=True)
    for i, label in enumerate(df['label']):
        try:
            df.at[i, 'label'] = label_unicode1[label]
        except KeyError:
            pass
    a = 5
    df.to_csv(path_or_buf='img_name_and_labels_unicode.csv', index=None, header=True, encoding='utf-8')
