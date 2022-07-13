from bs4 import BeautifulSoup
import requests
import re
from urllib.request import urlopen
import contextlib

source_code = requests.get('http://www.achemenet.com/en/tree/?/textual-sources/texts-by-publication')
soup = BeautifulSoup(source_code.content, 'lxml')
header = 'http://www.achemenet.com'
links = []
links2 = []
url_list = []

end_links = []
for link in soup.findAll('a', attrs={'href': re.compile("^/en/tree/\?/textual-sources/texts-by")}):
    link = link.get('href')
    if link.endswith('#set'):
        end_links.append(link[:-4])
    else:
        links.append(link)
print(len(end_links))
for i,link in enumerate(end_links):
    print(i)
    try:
        link = header+link+'/1/96/0#set'
        #link = "http://www.achemenet.com/en/tree/?/textual-sources/texts-by-publication/Strassmaier_Darius/1/96/0#set"
        with contextlib.closing(urlopen(link)) as page_code:

            #page_code = urlopen(link)
            # soup = BeautifulSoup(page_code, "html.parser").encode('UTF-8')
            # lines = soup.strip()
            soup = BeautifulSoup(page_code, 'html.parser')
            #source_code = requests.get(link)
            #soup = BeautifulSoup(source_code.content, 'lxml')
            soup = soup.find(id="items-pager-div")

            #page_code = urlopen(link)
            #mydivs = soup.find_all("div", {"class": "label"})
            pages = soup.find_all("li")
            sub_pages = len(pages)-3 # 16 because for each type there are 16 pages of links in the site
            # for name in texts_types:
            for page_number in range(1, sub_pages + 1):
                page_link = link[:-11] + "/%s/96/0" % (page_number)
                with contextlib.closing(urlopen(link)) as page_code:
                    #page_code = urlopen(page_link)

                    soup = BeautifulSoup(page_code, 'html.parser')
                    # Put spaces before and after so italic will always be a mydivs = {ResultSet: 1} [<ul class="item">\n<li class="info">\n<div class="info">\n<a name="set"></a>1 items\r\n          <fieldset>\n<input checked="checked" id="thumbnails-checkbox" type="checkbox" value="show"/>\n<label for="thumbnails-checkbox">show thumbnails</label>\n</fieldset>\n</… Viewword alone
                    #soup = soup.find(id="tree-structure-div")
                    mydivs = soup.find_all("div", {"class": "item"})
                    #result = soup.select('div.thumbnail-panel.document.location')

                    #a = mydivs[0].children
                    for div in mydivs:
                        k = div.find('a', href=True)['href']#[19:-2]
                        url_list.append(k)
    except:
        pass
    print(k)
with open('achemenet_page_urls.txt', 'w') as f:
    for item in url_list:
        f.write("%s\n" % item)