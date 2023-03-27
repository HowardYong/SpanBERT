# COMS E6111 - Advanced Database Systems Project 2

## Authors

Howard Yong (hy2724) and Solomon Chang (sjc2233)

## Files

Name | Usage
--- | ---
``README.pdf`` | README file
``query_transcripts_spanbert.pdf`` | Transcript of required spanbert query
``query_transcripts_gpt3.pdf`` | Transcript of required gpt3 query
``pytorch_pretrained_bert/`` | Pretrained spanbert files
``download_finetuned.sh`` | Spanbert setup file
``example_relations.py`` | Example of using spacy and spanbert
``project2.py`` | Main project file
``relation_set.py`` | Implements a data structure for a global relation set
``relations.txt`` | Contains a list of the relation names
``requirements.txt`` | A list of requirements to install
``setup.sh`` | Shell script to install project dependencies
``spacy_help_functions.py`` | Contains methods for relation extraction
``spanbert.py`` | Contains spanbert program

## Credentials

Credential | Detail
--- | ---
``AIzaSyBTMbRD_IajPp_IY1jVcwG2p2uv1Xe1dI4`` | Google API KEY
``485cb07d083282383`` | Engine ID

## Dependencies

To install, run:

  ```bash
  $ cd proj2
  $ bash setup.sh
  ```

## How to Run

Under the project's root directory, run

```bash
$ python3 project2.py [-spanbert|-gpt3] <google api key> <google engine id> <openai secret key> <relation> <threshold> <query> <k-tuples>
```

Examples:

```bash
$ python3 project2.py -spanbert  <google api key> <google engine id> <openai secret key> 1 0.7 "mark zuckerberg harvard" 10
```

## Internal Design

### Project Flow

The user provides an initial seed query, relation of interest, confidence threshold, and a desired number of relations to extract, k. The program is then responsible for launching a search with the provided query. A request is sent to each webpage response. If successful with respect to some criteria, then the main text is extracted from the contents of the webpage, preprocessed, and trimmed, if necessary. For each web page main text, a document is constructed using the `spaCy` library `en_core_web_lg` pre-trained language model for annotation and named entity recognition. 

Entity pairs are created for each sentence in the main text of a given web page document. If the 2 entity labels in a given pair conforms to the structure of the desired relation (provided upon program launch), then the project proceeds to relation extraction. Method of extracting relations is specified at program launch (`-spanbert` or `gpt3`). If the user provided the `-spanbert` flag, then the list of entity pairs are provided to the pre-trained SpanBERT model for relation extraction. Entity pairs consist of the corresponding context tokens, the subject (string, label, and index span found in the sentence), and object (same as subject). However, if the `-gpt3` flag is provided, then a prompt is constructed using a template, the input sentence, and an example relation. Relations that are extracted by SpanBERT have a corresponding confidence. If this confidence is higher than the user-provided threshold and is unique, it is added to the results. Relations extracted with GPT-3 are assigned a confidence of `1.0`. Thus, all unique relations are appended.

One iteration in this project flow consists of launching a search for a given query, downloading, annotating and extracting relations for each of the web pages returned for a given search. At the end of an iteration, the query is updated with an extracted relation based on 2 criteria: (1) it has the highest probability, or confidence (2) it is unique. If no more extracted relations can be used to construct a new query and the program has not terminated (i.e., `k` relations have not been extracted yet) then the program by default terminates as a result of iterative set expansion stalling. Otherwise, keep iterating until `k` relations have been extracted.

### Main functions

The key functions and objects in this project are listed below with short descriptions:

`main()`: Entry point and driver for the program

`search()`: Constructs search object with Google custom search API and launches search

`extract_content()` and `extract_main_text()`: Uses `BeautifulSoup4` to scrape webpage content, preprocess text, and return main text for annotation

`update_query()`: Searches extracted relations and updates query with next highest confidence, unique relation

`extract_relations()` and `extract_relations_gpt3()`: Applies pre-trained language model to extract relations

`create_entity_pairs()` and `create_entity_pairs_gpt3()`: Annotate text with `spaCy` library and create entity pairs

`RelationSet`: Custom class used to store relations. Handles duplicates and ordering with priority queue and set data structures.

### Details on scraping, annotation and relation extraction
#### Scraping

Dependencies: `google-api-python-client`, `requests`, `BeautifulSoup`, `re`

A search is launched using the `google-api-python-client`. This returns a JSON object storing each webpage (up to 10 as configured) and its corresponding metadata. Only non-PDF webpages are parsed

#### Annotation and Extraction

When running `-gpt3`, the program will loop through each sentence in the document and at the start of each iteration, makes a copy of the current state of the relation set for that document. SpaCY is then utilized to extract entity pairs out of the sentence. If the sentence contains entity pairs relevant to the desired relation, a plain text version of the sentence is fed into the GPT-3 model for relation extraction using this prompt

``` python
Please extract all the <Relation of Interest> relations from the sentence <Sentence>. 
Output Format: [<Subject>, <Relation of Interest>, <Object>]. 
Output Example: [<Example Output>]
```

The `Relation of Interest` is one of `Schools_Attended, Work_For, Live_In, Top_Member_Employees`.

The `Subject` and `Object` are one of `"ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"`

The `Example Output` is one of the following depending on the `Relation of Interest`

``` python
   1: '["Jeff Bezos", "Schools_Attended", "Princeton University"]',
   2: '["Alec Radford", "Work_For", "OpenAI"]',
   3: '["Mariah Carey", "Live_In", "New York City"]',
   4: '["Nvidia", "Top_Member_Employees", "Jensen Huang"]'
```

Several variations of this prompt and varying temperatures were tested. The prompt and temperature judged to return the largest number of relevant relations was selected. A temperature of `0.2` was selected. The Output Example was added to the prompt in the vein of one-shot learning to guide GPT-3 to providing more accurate relations. 


## External references



## Additional Information


