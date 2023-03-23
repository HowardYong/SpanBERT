#!/usr/bin/env python

"""
Project 2
COMS 6111 - Advanced Databases
Spring 2023

Implemented iterative set expansion for information extraction.
"""

__authors__ = ["Howard Yong", "Solomon Chang"]

import os
import sys
import argparse
import re

from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import spacy
from spanbert import SpanBERT
from spacy_help_functions import *
import openai

from collections import Counter
from collections import defaultdict
from pprint import pprint
import numpy as np
from numpy import dot
from numpy.linalg import norm



def search(google_api_key, google_engine_id, q):
    '''
    Launches instance of Google Programmable Search to query provided terms
    :params: 
    :return: 
    '''
    key = google_api_key
    searchEngineId = google_engine_id
    service = build(
        "customsearch", "v1", developerKey=key
    )

    res = (
        service.cse()
        .list(
            q=q,
            cx=searchEngineId,
        )
        .execute()
    )
    if int(res['searchInformation']['totalResults']) < 10:
        print("Fewer than 10 results overall. Terminating...")
        return None
    return res


def extract_main_text(soup):
    main_text = ''
    main_content = soup.find('main')
    if not main_content:
        return main_text
    
    for tag in main_content.find_all(["div", "table"], {"class": ["toc", "infobox"]}):
        tag.decompose()

    pragraphs = main_content.find_all('p')
    rem_char = 10000
    for p in pragraphs:
        if rem_char <= 0:
            break
        p_text = format_text(p.get_text())
        if len(p_text) > rem_char:
            main_text += p_text[:rem_char] + ' '
            rem_char = 0
        else:
            main_text += p_text + ' '
            rem_char -= len(p_text)
    return main_text


def format_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\t\r\n]+', ' ', text)
    return text.strip()


def print_parameters(args):
    relations_of_interest = ['Schools_Attended', 'Work_For', 'Live_In', 'Top_Member_Employees']
    parameters = {
        'Client key': args.google_api_key,
        'Engine key': args.google_engine_id,
        'OpenAI key': args.openai_api_key,
        'Method': 'spanbert' if args.spanbert else 'gpt-3',
        'Relation': relations_of_interest[args.r-1],
        'Threshold': args.t,
        'Query': args.q,
        '# of Tuples': args.k
    }

    print('\nParameters:')
    for k, v in parameters.items():
        print(f'{k:<15} = {v:>5}')
    print('Loading necessary libraries...')

def extract_relations_gpt3(sentence, openai_api_key, relation_type, threshold):
    openai.api_key = openai_api_key

    examples = {
        'Schools_Attended': '["Jeff Bezos", "Schools_Attended", "Princeton University"]',
        'Work_For': '["Alec Radford", "Work_For", "OpenAI"]',
        'Live_In': '["Mariah Carey", "Live_In", "New York City"]',
        'Top_Member_Employees': '["Jensen Huang", "Top_Member_Employees", "Nvidia"]'
    }

    # prompt = f"Given the following example of a relation: {examples[relation_type]}, please extract all the {relation_type} relations in the format of [\"Entity1\", \"{relation_type}\", \"Entity2\"] from the sentence: '{sentence}'. List all relevant relations."
    prompt = f"Given the following example of a relation: {examples[relation_type]}, please extract all the {relation_type} relations in the format of [\"Entity1\", \"{relation_type}\", \"Entity2\"] from the sentence: '{sentence}'. If the relation is not directly mentioned in the sentence, infer the correct one based on the context and the provided example. List all relevant relations."
    # prompt = f"Given the following example of a relation: {examples[relation_type]}, please extract all the {relation_type} relations in the format of [\"Entity1\", \"{relation_type}\", \"Entity2\"] from the sentence: '{sentence}'. If the relation is not directly mentioned in the sentence, infer the correct one based on the context and the provided example. When extracting education-related relations, consider different levels of education if applicable. List all relevant relations."

    print("my prompt is: ")
    print(prompt)
    print("END PROMPT\n")

    model = 'text-davinci-003'
    max_tokens = 100
    temperature = 0.2
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0

    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    # print("GPT RESPONSE IS: ", response)
    # print("PDSKLFJLSKDFJLSDKJFLSDJF")

    # result = response.choices[0].text.strip()
    relations_string = response.choices[0].text.strip()
    relations_string = relations_string.replace("\n", "")
    relations_string = '[' + relations_string + ']'
    print("GPT RESPONSE IS: ")
    print(relations_string)
    print("GPT RESPONSE END\n")
    # exit()
    try:
        # extracted_relation = ast.literal_eval(result)
        print("trying1")
        relations_list = ast.literal_eval(relations_string)
        print("trying2")
        print("EXTRACTED REL IS: ")
        print(relations_list)
        # print(type(extracted_relation))
        print("EXTRACTED REL  END\n")
        validated_relations = []
        for relation in relations_list:
            if len(relation) == 3 and relation[0] and relation[1] == relation_type and relation[2]:
                validated_relations.append(relation)
        print("validated REL IS: ")
        print(validated_relations)
        # print(type(extracted_relation))
        print("validated REL  END\n")
        return validated_relations
    except (SyntaxError, AssertionError, ValueError):
        return None



def main(args):
    print_parameters(args)
    entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    relation_to_entities = [
        [entities_of_interest[1], entities_of_interest[0]],
        [entities_of_interest[1], entities_of_interest[0]], 
        entities_of_interest[1:], 
        [entities_of_interest[0], entities_of_interest[1]], 
    ]
    nlp = spacy.load("en_core_web_lg")  

    if args.spanbert:
        model = SpanBERT("./pretrained_spanbert")  
    else:
        model = None
    X = set()
    visited = set()
    res = search(args.google_api_key, args.google_engine_id, args.q)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}

    n_iter = 0
    while len(X) < args.k:
        print(f'=========== Iteration: {n_iter} - Query: {args.q} ===========')
        for i in range(len(res['items'])):
            num_webpages = len(res['items'])
            webpage = res['items'][i]
            link = webpage['link']
            if webpage['link'] in visited:
                print('\tWebpage has already been visited. Skipping...')
                continue
            print(f'URL ({i+1} / {num_webpages}): {link}')
            visited.add(webpage['link'])

            print (f'\tFetching text from url...')
            response = requests.get(webpage['link'], headers=headers, timeout=20)
            if response.status_code != 200:
                print(f'\tWarning (response {response.status_code}): Target address {webpage["link"]}. Failed to retrieve webpage.')
                continue
            else:
                content = response.text
            soup = BeautifulSoup(content, 'html.parser')
            text = extract_main_text(soup)
            if len(text) == 0:
                print('\tWebpage has no main text to extract. Skipping...')
                continue
            
            doc = nlp(text)

            print('\tAnnotating the webpage using spacy...')
            relations, num_sentences_used = extract_relations(doc, model, relation_to_entities[args.r-1], args.t)


            print(f'\tExtracted annotations for  {num_sentences_used}  out of total  {len([s for s in doc.sents])}  sentences.')
            print(f'\tRelations extracted from this website: {len(relations)} (Overall: {len(X)})\n')
            print("Relations: {}".format(dict(relations)))
        n_iter += 1
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Project2',
        description='Information extraction for relations'
    )
    
    parser.add_argument('-spanbert', '--spanbert', action='store_true', help='Using SpanBERT for extraction')
    parser.add_argument('-gpt3', '--gpt3', action='store_true', help='Using GPT-3 for extraction')
    parser.add_argument('google_api_key', help='Google custom search JSON API key')
    parser.add_argument('google_engine_id', help='Google custom search engine ID')
    parser.add_argument('openai_api_key', help='OpenAI API key')
    parser.add_argument('r', choices=[1,2,3,4], type=int, help='Relation to extract')
    parser.add_argument('t', type=float, help='Extraction confidence threshold')
    parser.add_argument('q', help='Seed query provided as list of words')
    parser.add_argument('k', type=int, help='Number of tuples requested in output')
    args = parser.parse_args()
    main(args)
