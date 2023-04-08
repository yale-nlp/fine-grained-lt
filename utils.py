import wikipedia
import scispacy
import spacy
import requests
from bs4 import BeautifulSoup
import re
import nltk.data
import os

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

nlp_main = spacy.load("en_core_web_lg")
nlp_bio  = spacy.load("en_ner_bionlp13cg_md")

def remove_html_tags(text):
    """
    Remove html tags from a string
    Source: https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def extract_predicate(summary):
    first_sentence = sent_tokenizer.tokenize(summary)[0]
    predicate = ' '.join(first_sentence.split(' is ')[1:])
    return predicate

def extract_entities(text, bio):
    """_summary_

    Args:
        text (_type_): _description_
    """
    if bio:
        doc = nlp_bio(text)
    else:
        doc = nlp_main(text)
    return list(set(map(str, doc.ents)))

def search_wikipedia(term):
    """_summary_

    Args:
        term (_type_): _description_
    """
    result = wikipedia.search(term, results = 5) # get list of page names
    for r in result:
        try:
            summary = wikipedia.summary(r)
            return summary
        except:
            pass
    return ''

def clean_term(s):
    return s.lower().strip().replace('/','-').replace(' ','_')

def search_medline(term):
    """_summary_

    Args:
        term (_type_): _description_

    Returns:
        _type_: _description_
    """
    term_clean = clean_term(term)
    if f"{term_clean}.txt" in os.listdir("data/medline"):
        with open(f"data/medline/{term_clean}.txt") as f:
            lines = f.readlines()
        return ''.join(lines)
    try:
        url = f"https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term={term}"
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        documents = soup.find_all('document')
        # title = documents[0].find("content", {"name":"title"}).text
        summary = documents[0].find("content", {"name":"FullSummary"}).text
        # title = remove_html_tags(title)
        summary = remove_html_tags(summary)
    except:
        summary = ''
    
    file = open(f"data/medline/{term_clean}.txt", 'w')
    file.write(summary)
    file.close()

    return summary
    
def replace_entities(s, bio=False, entities=[]):
    # Extract entities
    entity_lst = extract_entities(s, bio) if len(entities)==0 else entities
    num_entities = len(entity_lst)
    
    # Search entity descriptions from Wiki/MedLine
    entity_dict = {key: search_medline(key) for key in entity_lst} if bio else \
        {key: search_wikipedia(key) for key in entity_lst}
    
    # For each entity, add in the context when possible
    num_replaced = 0
    for key in entity_dict:
        if entity_dict[key] != '':
            clean_ver = extract_predicate(entity_dict[key])
            if clean_ver != '':
                s = s.replace(key, f"{key} ({clean_ver})")
                num_replaced += 1
    
    # Return a flag indicating whether all, some, or none of the 
    # entities were replaced
    if num_entities > 0:
        if num_replaced == num_entities: flag = 'all'
        elif num_replaced > 0:           flag = 'some'
        else:                            flag = 'none'
    else: flag = 'none'          
    
    return s, flag

