# -*- coding: utf-8 -*-

from __future__ import print_function
import urllib
import os
import sys
import re
from bs4 import BeautifulSoup
from argparse import ArgumentParser
from collections import Counter
from urllib.request import urlopen
import pandas as pd
parser = ArgumentParser(description='Babylonian texts scraping from the site "Achemenet".')
parser.add_argument('--out', type=str, default='./achemenet_texts_20102019',
                    help='location of the output texts')
parser.add_argument('--sample', type=bool, default=False,
                    help='scrape only a sample from the texts')
args = parser.parse_args()


def user(path):
    """
    Get the user preferences.

    :param path: Path in which the file will be.
    :return: The printing preferences, 1 or 2.
    """
    choice = input("How would you like to see the process?\n"
                   "1 = by % \n"
                   "2 = printing the text \n")
    # Creating the directory for the files
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("You already got a directory with the same name in this directory", path)
        helper = input("Are you sure you want to replace the files 'texts.txt' and 'texts_urls.txt' in it?\n"
                       "No other files in that directory will be harmed.\n"
                       "1 - yes, I want to replace them\n"
                       "2 - no, I don't want to replace them\n")
        if helper != '1':
            exit()
    return choice


def print_progress(counter, end, view_choice, value):
    """
    Printing progress to user according to the choice.

    :param counter: What iteration are we in right now
    :param end: What is the last iteration
    :param view_choice: 1 or 2, How the user wants to see the process.
    :param value: What to print if 2 is chosen
    """
    if view_choice == '1':
        b = str(round((counter / end) * 100, 3))
        print('\r', end="")
        print(b + "%", end="")
    else:
        print(value)


def get_urls(path, view_choice, sample):
    """
    Copy the urls to a file in path.
    Printing the process according to the users choice

    :param path: The path to the new file.
    :param view_choice: 1 or 2, How the user wants to see the process.
    :return: A list with all the urls.
    """

    url_list = []
    with open(path, "w") as urls_text_file:
        percentage = 1.0
        '''texts_types = ["archives-ebabbar", "archives-eanna", "archives-egibi",
                       "archives-nappahu", "archives-murasu", "autres-archives-privees"]'''
        sub_pages = 2  # 16 because for each type there are 16 pages of links in the site
        if sample:
            sub_pages = 2
        #for name in texts_types:
        for page_number in range(1, sub_pages+1):
            page_link = "http://www.achemenet.com/fr/tree/?/sources-textuelles/" \
                        "textes-par-publication/Strassmaier_Cyrus/%s/24/0" % (page_number)

            page_code = urlopen(page_link)
            #soup = BeautifulSoup(page_code, "html.parser").encode('UTF-8')
            #lines = soup.strip()
            lines = page_code.readlines()
            for line in lines:
                line = str(line)
                if line.find("Strassmaier_Cyrus/166") >= 0 or line.find("Strassmaier_Cyrus/167") >= 0:
                    url_list.append("http://www.achemenet.com/" + line.split('"')[3])
                    urls_text_file.write("http://www.achemenet.com/" + line.split('"')[3] + "\n")
                    # There are 384 texts
                    print_progress(percentage, 384, view_choice, "http://www.achemenet.com/" + line.split('"')[3])
                    percentage += 1
    forbidden_urls = [
        "http://www.achemenet.com//fr/item/?/sources-textuelles/textes-par-langues-et-ecritures/babylonien/archives-ebabbar/1675180"]
    for item in forbidden_urls:
        if item in url_list:
            url_list.remove(item)
    return url_list


def clean_text(text, url):
    """
    Cleaning the given text.
    Using re.

    :param text: The texts needed to clean
    :param url: The url it came from
    :return: The cleaned text
    """
    try:
        # We don't like TRADUCTION in our text
        if not text.find("TRADUCTION") == -1:
            text = text.split("TRADUCTION",1)[0]
            print("\nTRADUCTION found, url:", url)
            #return ""

        # Cut the text in the right places
        text = text.split('TRANSLITTERATION')[1] \
            .split('BIBLIOGRAPHIE')[0] \
            .split('enregistrer dans')[0] \
            .split('REMARQUES')[0]

        # Turns numbers which are not in sub script to NUM
        '''text = re.sub(r'(?<!SUBSCRIPTSTART)[.0-9]+(?!SUBSCRIPTEND)', 'NUM', text)
        # Turn super script I to NAME
        text = re.sub(r'SUPSCRIPTSTARTISUPSCRIPTEND[^ ]+', 'NAME', text)
        text = re.sub(r'SUPSCRIPTSTARTIdSUPSCRIPTEND[^ ]+', 'NAME', text)
        # Turn super script f to FEMALENAME
        text = re.sub(r'SUPSCRIPTSTARTfSUPSCRIPTEND[^ ]+', 'FEMALENAME', text)
        # Turn super script KI to LOCATION
        text = re.sub(r'[^ ]+SUPSCRIPTSTARTkiSUPSCRIPTEND', 'LOCATION', text)
        # Turn super script URU to LOCATION
        text = re.sub(r'SUPSCRIPTSTARTuruSUPSCRIPTEND[^ ]+', 'LOCATION', text)
        # Turn super script iti to MONTH
        text = re.sub(r'SUPSCRIPTSTARTitiSUPSCRIPTEND[^ ]+', 'MONTH', text)
        # Turn super script d to GODNAME
        text = re.sub(r'SUPSCRIPTSTARTdSUPSCRIPTEND[^ ]+', 'GODNAME', text)'''
        for illegal_char in '+\'!?()§$*{}"':
            text = text.replace(illegal_char, "")
        # Turn super script "I" to "I "
        text = re.sub(r'SUPSCRIPTSTARTISUPSCRIPTEND', 'I ', text)
        # Turn super script Id to I d
        text = re.sub(r'SUPSCRIPTSTARTIdSUPSCRIPTEND', 'I d ', text)
        text = re.sub(r'SUPSCRIPTSTARTdSUPSCRIPTEND', 'd ', text)
        text = re.sub(r'd en', 'd+en', text)
        text = re.sub(r'd nà', 'd+nà', text)
        text = re.sub(r'd innin', 'd+innin', text)
        # Turn super script d to GODNAME
        text = re.sub(r'-', ' ', text)
        text = re.sub(r' ', " ", text)
        text = re.sub(r'\d ', "", text)
        text = re.sub(r',', ' ,', text)
        text = re.sub(r'../..', ' .', text)
        text = text.replace('[', "<")
        text = text.replace(']', ">")
        text = re.sub(r'<.*?>', '', text)
        #text = re.sub(r'<[*]> \n\n', '\n\n', text)

        text = text.replace('[', "")
        text = text.replace(']', "")
        text = text.replace('—', "")
        text = text.replace('>', "")
        text = text.replace('<', "")
        # Removing all of the illegal chars


        # for idx, char in enumerate(text):
        #     s = text.find('ITALICSTART', idx)
        #     e = text.find('ITALICEND', idx)
        #     if e == -1:
        #         break
        #     if idx < e < s or s == -1:
        #         if char == '-':
        #             text = text[:idx] + '+' + text[idx + 1:]

        # text = text.replace('+', '')

        # Replacing the 'x' and 'o' chars that means unknown words in the <UNK> token
        text = text.replace("x", " <UNK> ")
        text = text.replace(" o ", " <UNK> ")

        '''text = text.replace('SUPSCRIPTEND', '</sup>')
        text = text.replace('SUPSCRIPTSTART', '<sup>')
        text = text.replace('SUBSCRIPTEND', '</sub>')
        text = text.replace('SUBSCRIPTSTART', '<sub>')

        text = text.replace('ITALICSTART', ' <i> ')
        text = text.replace('ITALICEND', ' </i> ')'''

        text = text.replace('SUPSCRIPTEND', ' ')
        text = text.replace('SUPSCRIPTSTART', ' ')
        text = text.replace('SUBSCRIPTEND', '')
        text = text.replace('SUBSCRIPTSTART', '')

        text = text.replace('ITALICSTART', '')
        text = text.replace('ITALICEND', '')

        # Remove et = and
        text = text.replace('et', '')

        # Replacing all the unicode chars
        # Weird -
        #text = text.replace(u'\u2014', " <BRK> ")
        # Three dots but in one char
        text = text.replace(u'\u2026', " <BRK> ")
        # 'Round up' chars
        text = text.replace(u'\u2309', "")
        text = text.replace(u'\u2308', "")
        # ’ \u2019
        text = text.replace(u'’', "")

        text = text.replace("<sup></sup>", " ")
        text = text.replace("<sub></sub>", "")
        text = text.replace("<i>  </i>", "")
        text = text.replace("<i>   </i>", "")
        text = text.replace("<i>    </i>", "")


        #text = re.sub(r' <.*?>', "", text)

        text = text.replace(" ", " ")

        # Handling double <BRK> tokens
        '''for number in range(10, 1, -1):
            text = text.replace(" <BRK> " * number, " <BRK> ")'''

        # Removing lines that are just one number or empty
        text = text.replace("<BRK>", "")
        text = text.replace("<UNK>", "")
        '''line = [line for line in text.split('\n')]
        line_len =[l.strip() for l in line]'''

        text = '\n'.join([line for line in text.split('\n')
                          if not (line.strip().isdecimal() or line.strip() == '')])
        '''split_text = text.split('\n')
        strip_text = [line.strip() for line in split_text]'''
        text = text.replace(".0", "")
        text = text.replace(",0", "")
        text = text.replace(" 0 ", " ")
        text = re.sub('/', '%', text)
        #text = text.replace('.', " .")

    except Exception as e:
        print("\n-------\n"
              "Problem cleaning the text from the url:", url, "\n"
                                                              "Exception on line", str(sys.exc_info()[2].tb_lineno),
              ":\n",
              e,
              "\n-------\n")
        # If failed then return an empty text
        text = ""
    return text


def parse_italic(input_path, output_path, view_choice):
    with open(input_path, 'r') as inf, open(output_path, 'w') as outf:
        italic = False
        for line in inf.readlines():
            for word in line.split():
                if word == "<i>":
                    italic = True
                    continue
                elif word == "</i>":
                    italic = False
                    continue

                if word.startswith("http"):
                    italic = False

                if italic:
                    outf.write("<i>" + word + "</i> ")
                else:
                    outf.write(word + " ")
            outf.write('\n')


def parse_unk(input_path, output_path, thresh_hold):
    with open(input_path, 'r') as inf, open(output_path, 'w') as outf:
        counts = Counter([word for line in inf for word in line.split()])
    with open(input_path, 'r') as inf, open(output_path, 'w') as outf:
        for line in inf.readlines():
            for word in line.split():
                if word.startswith("http"):
                    outf.write(word + " ")
                    continue
                if counts[word] > thresh_hold:
                    outf.write(word + " ")
                else:
                    outf.write("<UNK>" + " ")
            outf.write('\n')


def get_texts(path, view_choice, url_list):
    """
    Getting the texts and cleaning them, prints the results to the file in path.

    :param path: Path to print all the tests to.
    :param view_choice: 1 or 2, How the user wants to see the process.
    :param url_list: The list of all the urls.
    """

    with open(path, "w", encoding="utf-8") as texts_file:
        percentage = 1.0
        df = pd.DataFrame()
        for url in url_list:
            # Get the texts from the site, parse and clean them

            # Get the texts from the site + parsing
            try:
                page_code = urlopen(url)
                soup = BeautifulSoup(page_code, 'html.parser')
                soup = str(soup)

                # Put spaces before and after so italic will always be a word alone
                soup = soup.replace('<i>', '')
                soup = soup.replace('</i>', '')

                soup = soup.replace('</sup>', '')
                soup = soup.replace('<sup>', '')
                soup = soup.replace('</sub>', '')
                soup = soup.replace('<sub>', '')

                soup = BeautifulSoup(soup, 'html.parser')
                title = ['Title:' + soup.find(id="item-note-title-div").text.strip()]
                headers = title + soup.find(id="item-note-headers-div").text.strip().split('\n')
                clean_headers = [re.sub('\s+', ' ', line) for line in headers
                                 if (len(line.strip())>1 and not line.startswith('ITALIC'))]
                fix_traduction_header = []
                i =0
                while i in range(len(clean_headers)):
                    if clean_headers[i].strip().endswith(':'):
                        header = clean_headers[i]+clean_headers[i+1]
                        i+=2
                    else:
                        header = clean_headers[i]
                        i+=1
                    fix_traduction_header.append(header)

                headers_key = [line.split(':') for line in fix_traduction_header]
                headers_key = [line.split(':')[0] for line in fix_traduction_header]
                headers_val = [line.split(':')[1] for line in fix_traduction_header]
                d = {key:val for key,val in zip(headers_key,headers_val)}
                df_new = pd.DataFrame(data=d, index=[0])
                df = pd.concat([df, df_new], sort=False, ignore_index=True).fillna('NaN')
                a = 5
                '''df = pd.DataFrame(headers_val,columns=headers_key)
                df[headers_key] = headers_val
                for i, key in enumerate(headers_key):
                    df[key] = headers_val[i]'''
                #text = soup.get_text()
            except Exception as e:
                print("\n-------\n"
                      "Problem getting or parsing the url:", url, "\n"
                                                                  "Exception:\n",
                      e,
                      "\n-------\n")
                continue

            # Texts Cleaning
            '''cleaned_text = clean_text(text, url)
            if cleaned_text == "":
                continue
            #a = cleaned_text.encode("utf-8")
            print_progress(percentage, len(url_list), view_choice, url + '\n' + cleaned_text + '\n')
            percentage += 1

            # Writing the url and the text to the file
            texts_file.write(url + '\n' + cleaned_text + '\n\n')'''
        df.to_csv(path_or_buf='cyrus_metadata.csv', index=None, header=True)
        a = 5

if __name__ == '__main__':
    """
    Texts scraping from the site "Achemenet".
    Written By: Yonatan Lifshitz.
    The downloading if for Mac / Linux filing system.

    Enter the number according to the way you want to see the printing.

    After the process will finish you will find a new directory,
    in the directory in which you ran the code, named: achemenet_texts.
    This directory contains two files: 
    texts.txt - contains all the texts after scraping separated by the fitting url.
    texts_urls.txt - contains all the texts urls.
    """

    print('Babylonian texts scraping from the site "Achemenet".\n')

    main_path = args.out
    if args.sample:
        print('Notice that you are in "sample" mode.')
    #user_view_choice = user(main_path)

    url_path = main_path + "/texts_urls.txt"
    texts_url = main_path + "/texts.txt"

    print("\nStarting urls download to: " + url_path)
    urls = get_urls(url_path, 1, args.sample)
    print("\nFinished download the urls")

    print("\nStarting texts download to: " + texts_url)
    get_texts(main_path + "/texts.txt", 1, urls)
    print("\nFinished downloading texts to: " + texts_url)

    '''print("\nParsing italics: " + main_path + "/italic_texts.txt")
    parse_italic(main_path + "/texts.txt", main_path + "/italic_texts.txt", user_view_choice)

    print("\nParsing unk: " + main_path + "/unk_texts.txt")
    parse_unk(main_path + "/italic_texts.txt", main_path + "/unk_texts.txt", 2)'''

    print("\nDONE\n")
