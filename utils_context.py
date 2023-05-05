from bs4 import BeautifulSoup
from nltk.corpus import wordnet as wn
import nltk.data
import os
import openai
import re
import requests
import scispacy
import spacy
import wikipedia

REFERENCE_PATHS = {
    "wordnet_wikipedia": "data/wordnet_wikipedia",
    "medline": "data/medline",
    "mesh": "data/mesh",
    "umls": "data/umls",
}

sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


def gpt_simplify(
    text,
    prompt="simplify this text for a second-grade student:",
    temperature=0.2,
    max_length=256,
):
    """Use GPT 3.5 API to get the simplified result of a text"""
    augmented_prompt = prompt + text
    # Note: copy and paste your own openai secret key for testing, and don't upload it to public github repo.
    # the api key can also be saved in an environment variable and use os.getenv()
    openai.api_key = "copy and paste your openai api key here"

    try:
        output = openai.Completion.create(
            model="text-davinci-003",
            prompt=augmented_prompt,
            temperature=temperature,
            max_tokens=max_length,
        )

        return output["choices"][0]["text"]

    except:
        print("there is an error")


def remove_html_tags(text):
    """
    Remove html tags from a string
    Source: https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
    """
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def cut_desc_to_first_sentence(summary):
    first_sentence = sent_tokenizer.tokenize(summary)[0]
    if " is " in first_sentence:
        first_sentence = " ".join(first_sentence.split(" is ")[1:])
    return first_sentence


# def extract_entities(text, bio):
#     """_summary_

#     Args:
#         text (_type_): _description_
#     """
#     if bio:
#         doc = nlp_bio(text)
#     else:
#         doc = nlp_main(text)
#     return list(set(map(str, doc.ents)))


def clean_term(s):
    return s.strip().replace("/", "-").replace(" ", "_").replace(".", "_")


def clean_description(s):
    tags = [
        "<p>",
        "</p>",
        "<b>",
        "</b>",
        "<i>",
        "</i>",
        "<ul>",
        "</ul>",
        "<li>",
        "</li>",
    ]
    for tag in tags:
        s = s.replace(tag, "")
    return s


def search_wikipedia(term):
    """_summary_

    Args:
        term (_type_): _description_
    """
    # Get list of page names
    result = wikipedia.search(term, results=2)
    for r in result:
        try:
            summary = wikipedia.summary(r)
            if "may refer to:" in summary:
                continue
            return summary
        except:
            pass
    return ""


def search_medline(term):
    """_summary_

    Args:
        term (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        url = f"https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term={term}"
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")
        documents = soup.find_all("document")
        # title = documents[0].find("content", {"name":"title"}).text
        summary = documents[0].find("content", {"name": "FullSummary"}).text
        # title = remove_html_tags(title)
        summary = remove_html_tags(summary)
    except:
        summary = ""

    return summary


def search_history(term, kb):
    with open(f"{REFERENCE_PATHS[kb]}/{term}.txt") as f:
        lines = f.readlines()
    result = "".join(lines)
    if result is None:
        return ""
    else:
        return result


def search_wordnet(term):
    synset_lst = wn.synsets(term.replace(" ", "_"), pos=wn.NOUN)
    if len(synset_lst) > 0:
        return synset_lst[0].definition()
    else:
        return ""


# def add_context(s, kb, entities=[]):
def add_context(s, ner_model, kb, linker):

    # Extract entities
    entity_lst = set(ner_model(s).ents)
    num_entities = len(entity_lst)

    # Fetch search history for the kb
    existing_files = os.listdir(REFERENCE_PATHS[kb])

    # Search entity descriptions, but pull from history when possible
    if kb == "wordnet_wikipedia":
        entity_dict = {}
        entity_lst = list(
            filter(
                lambda e: e.label_
                in [
                    "EVENT",
                    "FAC",
                    "GPE",
                    "LANGUAGE",
                    "LAW",
                    "LOC",
                    "NORP",
                    "ORG",
                    "PERSON",
                    "PRODUCT",
                    "WORK_OF_ART",
                ],
                entity_lst,
            )
        )
        for e in entity_lst:
            if f"{clean_term(str(e))}.txt" in existing_files:
                entity_dict[str(e)] = search_history(
                    clean_term(str(e)), "wordnet_wikipedia"
                )
            else:
                desc = search_wordnet(str(e))
                if desc == "":
                    desc = search_wikipedia(str(e))
                entity_dict[str(e)] = desc
    elif kb == "medline":
        entity_dict = {
            key: search_medline(key)
            if f"{clean_term(key)}.txt" not in existing_files
            else search_history(clean_term(key), "medline")
            for key in list(map(str, entity_lst))
        }
    elif kb in ["umls", "mesh"]:
        entity_dict = {}
        for e in entity_lst:
            if f"{clean_term(str(e))}.txt" in existing_files:
                entity_dict[str(e)] = search_history(clean_term(str(e)), kb)
            elif len(e._.kb_ents) > 0:
                umls_ent_id, score = e._.kb_ents[0]  # Get top hit from UMLS
                entity_dict[str(e)] = (
                    linker.kb.cui_to_entity[umls_ent_id][4] if score >= 0.75 else ""
                )  # Use threshold of 0.9 to filter out legit descriptions
            else:
                entity_dict[str(e)] = ""
    else:
        assert False, print("kb must be in ['wikipedia','medline','umls','mesh']")

    # Ensure the entity_dict does not have Null values and clean the description
    for key in entity_dict:
        if entity_dict[key] is None:
            entity_dict[key] = ""
        entity_dict[key] = clean_description(entity_dict[key])

    # Save for future reference
    for key in entity_dict:
        cleaned_term = clean_term(key)
        if f"{cleaned_term}.txt" not in existing_files:
            file = open(f"{REFERENCE_PATHS[kb]}/{cleaned_term}.txt", "w")
            file.write(entity_dict[key])
            file.close()

    # For each entity, add in the context when possible
    num_replaced = 0
    for key in entity_dict:
        if entity_dict[key] != "":
            description = cut_desc_to_first_sentence(entity_dict[key])
            assert description != ""
            s = s.replace(key, f"{key} ({description})")
            num_replaced += 1

    # Return a flag indicating whether all, some, or none of the
    # entities were replaced
    if num_entities > 0:
        if num_replaced == num_entities:
            flag = "all"
        elif num_replaced > 0:
            flag = "some"
        else:
            flag = "none"
    else:
        flag = "none"

    return s, flag