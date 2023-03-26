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
    Launches instance of Google Programmable Search to query provided terms.
    :params: str, str, str
    :return: JSON
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
    if int(res['searchInformation']['totalResults']) < 1:
        print("Returned 0 results. Terminating...")
        return None
    return res


def extract_content(webpage):
    '''
    Retrieve content from webpage if response status is OK. 
    :params: dict
    :return: BeautifulSoup, None
    '''
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}
    print (f'\tFetching text from url...')
    response = requests.get(webpage['link'], headers=headers, timeout=20)
    
    if response.status_code != 200:
        print(f'\tWarning (response {response.status_code}): Target address {webpage["link"]}. Failed to retrieve webpage.')
        return None
    else:
        content = response.text
    soup = BeautifulSoup(content, 'html.parser')
    return soup


def extract_main_text(soup):
    '''
    Extract main text from webpage content. If number of characters exceeds 10000 then truncate.
    :params: BeautifulSoup
    :return: str
    '''
    main_text = ''
    paragraphs = soup.find_all()

    rem_char = 10000
    for p in paragraphs:
        if rem_char <= 0:
            break
        p_text = format_text(p.get_text())
        if len(p_text) > rem_char:
            main_text += p_text[:rem_char] + ' '
            rem_char = 0
        else:
            main_text += p_text + ' '
            rem_char -= len(p_text)
    
    if len(soup.get_text(strip=True)) > 10000:
        print(f'\tTrimming webpage content from {len(soup.get_text(strip=True))} to 10000 characters')
        print(f'\tWebpage length (num of characters): ', len(main_text[:10000]))
    else:
        print(f'\tWebpage length (num of characters): ', len(soup.get_text(strip=True)))
    if len(main_text) == 0:
        print('\tWebpage has no main text to extract. Skipping...')
        return ''
    return main_text[:10000]


def format_text(raw_text_str):
    '''
    Preprocess text extracted from webpage content.
    :params: str
    :return: str
    '''
    preprocessed_text = re.sub(u'\xa0', ' ', raw_text_str) 
    preprocessed_text = re.sub('\t+', ' ', preprocessed_text) 
    preprocessed_text = re.sub('\n+', ' ', preprocessed_text) 
    preprocessed_text = re.sub(' +', ' ', preprocessed_text) 
    preprocessed_text = preprocessed_text.replace('\u200b', '')
    return preprocessed_text.strip()


def print_parameters(args):
    '''
    Displays header with argument values.
    :params: argparse.Namespace
    :return: void
    '''
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
        print(f'{k:<15} = {v:>2}')
    print('Loading necessary libraries...')


def update_query(X, old_queries=None):
    '''
    Updates with unique query by iterating extracted relations in descending order of confidence. 
    :params: RelationSet, set
    :return: str, None
    '''
    for i in range(len(X)):
        extracted_rel = (X[i][1][0] + ' ' + X[i][1][-1]).lower()
        if extracted_rel not in old_queries:
            return extracted_rel
    print('\nIterative set expansion has stalled retrieving k high-confidence tuples. Terminating...')
    return None


def main(args):
    # (1) Initialize empty RelationSet object and empty sets for visited webpages and old queries.
    print_parameters(args)
    n_iter = 0
    nlp = spacy.load("en_core_web_lg")  
    extractor = 'spanbert' if args.spanbert else 'gpt3'
    X = RelationSet(relation_type=args.r, model=extractor)
    visited = set()
    old_queries = set()
    query =  args.q
    old_queries.add(query)

    if args.spanbert:
        model = SpanBERT("./pretrained_spanbert")  
    else:
        model = None

    # (2) Obtain search results for given query until any-K (or top-K) tuples extracted.
    while len(X) < args.k and query:
        res = search(args.google_api_key, args.google_engine_id, query)
        num_webpages = len(res['items'])
        print(f'=========== Iteration: {n_iter} - Query: {query} ===========\n')

        # (3) Process each webpage.
        for i in range(num_webpages):
            # (3a) Only process unvisited webpages and non-PDF webpage content.
            webpage = res['items'][i]
            link = webpage['link']

            print(f'URL ({i+1} / {num_webpages}): {link}')
            if webpage['link'] in visited:
                print('\tWebpage has already been visited. Skipping...\n')
                continue
            elif 'fileFormat' in webpage and webpage['fileFormat'] == 'PDF/Adobe Acrobat':
                print('\tEncountered non-HTML webpage (PDF). Skipping...\n')
                continue
            visited.add(webpage['link'])

            # (3b-c) Extract main content with BeautifulSoup and get main text.
            content = extract_content(webpage)
            if content:
                text = extract_main_text(content)
                if text:
                    doc = nlp(text)
            else:
                continue

            print('\tAnnotating the webpage using spacy...')
            # (3d-e) Annotate with spaCy library and extract relations via method specified in args.
            if args.spanbert:
                relations, num_sentences_used, overall_num_relations = extract_relations(doc, model, args.r, args.t)
            else:
                relations, num_sentences_used, overall_num_relations = extract_relations_gpt3(doc, args.openai_api_key, args.r, args.t)
            
            # (3f-4) Update RelationSet with extracted relations. Skip duplicate relations.
            num_dup = 0
            for r, conf in relations.items():
                num_dup += X.add(r, conf)

            print(f'\tExtracted annotations for  {num_sentences_used}  out of total  {len([s for s in doc.sents])}  sentences.')
            print(f'\tRelations extracted from this website: {len(relations) - num_dup} (Overall: {overall_num_relations})\n')
        
        # (5-6) If at least k relations added to RelationSet X then exit loop. 
        # Otherwise update the query with the highest confidence extracted relation (for SpanBERT). 
        # If no unique query exists in X, terminate with ISE stalled.
        n_iter += 1
        query = update_query(X, old_queries)
    
    print(f'\n================== ALL RELATIONS for {[rel for rel in X.relation]} ( {len(X)} ) =================\n')
    print(X, '\n')
    print(f'Total # of iterations = {n_iter}\n\n')
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

