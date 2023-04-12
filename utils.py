import wikipedia
import scispacy
import spacy
import requests
from bs4 import BeautifulSoup
import re
import nltk.data
import textstat
import os
import openai

sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

nlp_main = spacy.load("en_core_web_lg")
nlp_bio = spacy.load("en_ner_bionlp13cg_md")


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


def get_readability_score(text, metric="flesch_reading_ease"):
    """get the readability score and grade level of text"""
    if metric == "flesch_reading_ease":
        score = textstat.flesch_reading_ease(text)
        if score > 90:
            grade = "5th grade"
        elif score > 80:
            grade = "6th grade"
        elif score > 70:
            grade = "7th grade"
        elif score > 60:
            grade = "8th & 9th grade"
        elif score > 50:
            grade = "10th to 12th grade"
        elif score > 30:
            grade = "college"  # Collge student = average 13th to 15th grade
        elif score > 10:
            grade = "college graduate"
        else:
            grade = "professional"
        return score, grade

    elif metric == "flesch_kincaid_grade":
        score = textstat.flesch_kincaid_grade(
            text
        )  # Note: this score can be negative like -1
        grade = round(score)
        if grade > 16:
            grade = "college graduate"  # Collge graduate: >16th grade
        elif grade > 12:
            grade = "college"
        elif grade <= 4:
            grade = "4th grade or lower"
        else:
            grade = f"{grade}th grade"
        return score, grade

    # elif metric == 'SMOG': # Note: SMOG formula needs at least three ten-sentence-long samples for valid calculation
    #     score = textstat.smog_index(text)
    #     grade = round(score)
    #     if grade > 16:
    #         grade = 'college graduate'
    #     elif grade > 12:
    #         grade = 'college'
    #     else:
    #         grade = f"{grade}th grade"
    #     return score, grade

    elif metric == "dale_chall":
        score = textstat.dale_chall_readability_score(text)
        if score >= 10:
            grade = "college graduate"
        elif score >= 9:
            grade = "college"  # Collge student = average 13th to 15th grade
        elif score >= 8:
            grade = "11th to 12th grade"
        elif score >= 7:
            grade = "9th to 10th grade"
        elif score >= 6:
            grade = "7th to 8th grade"
        elif score >= 5:
            grade = "5th to 6th grade"
        else:
            grade = "4th grade or lower"
        return score, grade

    elif metric == "gunning_fog":
        score = textstat.gunning_fog(text)
        grade = round(score)
        if grade > 16:
            grade = "college graduate"
        elif grade > 12:
            grade = "college"
        elif grade <= 4:
            grade = "4th grade or lower"
        else:
            grade = f"{grade}th grade"
        return score, grade

    else:
        raise ValueError(f"Unknown metric {metric}")


def remove_html_tags(text):
    """
    Remove html tags from a string
    Source: https://medium.com/@jorlugaqui/how-to-strip-html-tags-from-a-string-in-python-7cb81a2bbf44
    """
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def extract_predicate(summary):
    first_sentence = sent_tokenizer.tokenize(summary)[0]
    predicate = " ".join(first_sentence.split(" is ")[1:])
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
    result = wikipedia.search(term, results=5)  # get list of page names
    for r in result:
        try:
            summary = wikipedia.summary(r)
            return summary
        except:
            pass
    return ""


def clean_term(s):
    return s.lower().strip().replace("/", "-").replace(" ", "_")


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
        return "".join(lines)
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

    file = open(f"data/medline/{term_clean}.txt", "w")
    file.write(summary)
    file.close()

    return summary


def replace_entities(s, bio=False, entities=[]):
    # Extract entities
    entity_lst = extract_entities(s, bio) if len(entities) == 0 else entities
    num_entities = len(entity_lst)

    # Search entity descriptions from Wiki/MedLine
    entity_dict = (
        {key: search_medline(key) for key in entity_lst}
        if bio
        else {key: search_wikipedia(key) for key in entity_lst}
    )

    # For each entity, add in the context when possible
    num_replaced = 0
    for key in entity_dict:
        if entity_dict[key] != "":
            clean_ver = extract_predicate(entity_dict[key])
            if clean_ver != "":
                s = s.replace(key, f"{key} ({clean_ver})")
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
