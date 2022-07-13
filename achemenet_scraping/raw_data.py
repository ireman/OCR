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
import contextlib

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
    '''choice = input("How would you like to see the process?\n"
                   "1 = by % \n"
                   "2 = printing the text \n")'''
    choice = '1'
    # Creating the directory for the files
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print("You already got a directory with the same name in this directory", path)
        '''helper = input("Are you sure you want to replace the files 'texts.txt' and 'texts_urls.txt' in it?\n"
                       "No other files in that directory will be harmed.\n"
                       "1 - yes, I want to replace them\n"
                       "2 - no, I don't want to replace them\n")'''
        helper = '1'
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
        sub_pages = 16  # 16 because for each type there are 16 pages of links in the site
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

        # Cut the text in the right places
        text = text.split('TRANSLITTERATION')[1] \
            .split('BIBLIOGRAPHIE')[0] \
            .split('enregistrer dans')[0] \
            .split('REMARQUES')[0]

        # Turns numbers which are not in sub script to NUM

        text = re.sub(r' ', " ", text)
        text = re.sub(r'\d ', "", text)

        '''text = text.replace('SUPSCRIPTEND', '</sup>')
        text = text.replace('SUPSCRIPTSTART', '<sup>')
        text = text.replace('SUBSCRIPTEND', '</sub>')
        text = text.replace('SUBSCRIPTSTART', '<sub>')
        text = text.replace('ITALICSTART', '<i>')
        text = text.replace('ITALICEND', '</i>')'''

        # Remove et = and
        text = text.replace('et', '')

        # Replacing all the unicode chars
        # Weird -
        #text = text.replace(u'\u2014', " <BRK> ")
        # Three dots but in one char
        text = text.replace(u'\u2026', "")
        # 'Round up' chars
        text = text.replace(u'\u2309', "")
        text = text.replace(u'\u2308', "")
        # ’ \u2019
        text = text.replace(u'’', "")

        text = text.replace(" ", " ")


        text = '\n'.join([line for line in text.split('\n')
                          if not (line.strip().isdecimal() or line.strip() == '' or line.strip() == "<i> </i>" or line.strip() == "<i></i>"
                                  'Collations' in line.strip() or 'érasée' in line.strip() or 'ligne' in line.strip() or not re.search('[a-zA-Z0-9]', line.strip()))])


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
    header = 'http://www.achemenet.com'
    with open("output1.html", "wb") as file:
    #with open(path, "w", encoding="utf-8") as texts_file:
        percentage = 1.0
        df = pd.DataFrame()
        for url in url_list:
            # Get the texts from the site, parse and clean them
            url = header+url
            # Get the texts from the site + parsing
            try:
                page_code = urlopen(url)
                soup = BeautifulSoup(page_code, "html")
                soup = soup.find(id="item-div")
                html = soup.prettify("utf-8")
                file.write(html)
                # Put spaces before and after so italic will always be a word alone
                soup = soup.find(id="item-div")
                #soup = str(soup)
                '''soup = soup.replace('<i>', 'ITALICSTART')
                soup = soup.replace('</i>', 'ITALICEND')

                soup = soup.replace('</sup>', 'SUPSCRIPTEND')
                soup = soup.replace('<sup>', 'SUPSCRIPTSTART')
                soup = soup.replace('</sub>', 'SUBSCRIPTEND')
                soup = soup.replace('<sub>', 'SUBSCRIPTSTART')'''
                '''soup = soup.replace('<i>', '')
                soup = soup.replace('</i>', '')

                soup = soup.replace('</sup>', ' ')
                soup = soup.replace('<sup>', ' ')
                soup = soup.replace('</sub>', '')
                soup = soup.replace('<sub>', '')'''
                #soup = BeautifulSoup(soup, 'html.parser')
                text = soup.get_text()
            except Exception as e:
                print("\n-------\n"
                      "Problem getting or parsing the url:", url, "\n"
                                                                  "Exception:\n",
                      e,
                      "\n-------\n")
                continue

            # Texts Cleaning
            cleaned_text = clean_text(text, url)
            if cleaned_text == "":
                continue
            #a = cleaned_text.encode("utf-8")
            print_progress(percentage, len(url_list), view_choice, url + '\n' + cleaned_text + '\n')
            percentage += 1

            # Writing the url and the text to the file
            texts_file.write(url + '\n' + cleaned_text + '\n\n')


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
    user_view_choice = user(main_path)

    url_path = main_path + "/texts_urls.txt"
    texts_url = main_path + "/texts.txt"
    with open('achemenet_page_urls.txt', 'r') as f:
        urls = f.readlines()
    header = 'http://www.achemenet.com'
    i = 1
    for url in urls:
        url = header+url
        name = url.split('/')[-2]
        with open("all_html_file\\"+name+"_"+str(i)+".html", "w",encoding="utf_8") as file:
            try:
                with contextlib.closing(urlopen(url)) as page_code:
                #page_code = urlopen(url)
                    soup = BeautifulSoup(page_code, "lxml")
                    soup = soup.find(id="item-div")
                    html = soup.prettify("utf-8")
                    file.write(html)
                    i+=1
                    if i%10 == 0:
                        print(i)
            except:
                print("problem with url:"+ url)
    #get_texts(main_path + "/raw_texts.txt", user_view_choice, urls)
    #print("\nFinished downloading texts to: " + texts_url)

    '''print("\nParsing italics: " + main_path + "/italic_texts.txt")
    parse_italic(main_path + "/texts.txt", main_path + "/italic_texts.txt", user_view_choice)

    print("\nParsing unk: " + main_path + "/unk_texts.txt")
    parse_unk(main_path + "/italic_texts.txt", main_path + "/unk_texts.txt", 2)'''

    print("\nDONE\n")
