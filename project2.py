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

from spanbert import SpanBERT
from spacy_help_functions import *
from relation_set import RelationSet
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup
import spacy
import openai

from collections import Counter
from collections import defaultdict


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


def main(args):
    print_parameters(args)
    nlp = spacy.load("en_core_web_lg")  

    if args.spanbert:
        model = SpanBERT("./pretrained_spanbert")  
    else:
        model = None
    X = RelationSet()
    visited = set()
    res = search(args.google_api_key, args.google_engine_id, args.q)
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}

    n_iter = 0
    while len(X) < args.k:
        print(f'=========== Iteration: {n_iter} - Query: {args.q} ===========')
        for i in range(len(res['items'])):
        # for i in range(1):
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
            relations, num_sentences_used, overall_num_relations = extract_relations(doc, model, args.openai_api_key, args.r, args.t)
            for r, conf in relations.items():
                X.add(r, conf)
            print('[main, len(X), X]: ', len(X), X)

            print(f'\tExtracted annotations for  {num_sentences_used}  out of total  {len([s for s in doc.sents])}  sentences.')
            print(f'\tRelations extracted from this website: {len(relations)} (Overall: {overall_num_relations})\n')
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

