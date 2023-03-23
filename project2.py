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


def extract_main_content(soup):
    main_content_selectors = [
        {'tag': 'div', 'class': 'main-content'},
        {'tag': 'div', 'class': 'content'},
        {'tag': 'article'},
        {'tag': 'main'},
        {'tag': 'section', 'class': 'post-content'},
        {'tag': 'section', 'class': 'article-content'},
    ]

    for selector in main_content_selectors:
        if 'class' in selector:
            main_content = soup.find(selector['tag'], {'class': selector['class']})
        else:
            main_content = soup.find(selector['tag'])

        if main_content is not None:
            return main_content

    return soup


def format_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    # text = re.sub(r'[\t\r\n]+', ' ', text).strip()
    if len(text) > 10000:
        print(f'Trimming webpage content from {len(text)} to 10000...')
        return text[:10000]
    return text[:10000]


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


def main(args):
    print_parameters(args)
    entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    relations_of_interest = ['Schools_Attended', 'Work_For', 'Live_In', 'Top_Member_Employees']
    relation_entities = {
        'Schools_Attended': entities_of_interest[:2], 
        'Work_For': entities_of_interest[:2], 
        'Live_In': entities_of_interest[1:], 
        'Top_Member_Employees': entities_of_interest[:2]
    }
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
                continue
            print(f'URL ({i+1} / {num_webpages}): {link}')
            visited.add(webpage['link'])

            print (f'\tFetching text from url...')
            response = requests.get(webpage['link'], headers=headers, timeout=20)
            if response.status_code != 200:
                print(f'\tWarning (response {response.status_code}): Target address {webpage["link"]}. Failed to retrieve webpage.')
                continue
            else:
                content = response.content
            soup = BeautifulSoup(content, 'html.parser')
            soup = extract_main_content(soup)
            text = soup.get_text(strip=True)
            text = format_text(text)
            doc = nlp(text)

            print('\tAnnotating the webpage using spacy...')
            relations, num_sentences_used = extract_relations(doc, model, relation_entities[relations_of_interest[args.r-1]], args.t)

            print(f'\tExtracted annotations for  {num_sentences_used}  out of total  {len([s for s in doc.sents])}  sentences.')
            print(f'\tRelations extracted from this website: {len(relations)} (Overall: {len(X)})\n')
            print("Relations: {}".format(dict(relations)))
        n_iter += 1
        break
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

