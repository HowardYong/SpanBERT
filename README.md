# COMS E6111 - Advanced Database Systems Project 2

## Authors

Howard Yong (hy2724) and Solomon Chang (sjc2233)

## Files

Name | Usage
--- | ---
``README.pdf`` | README file
``spanbert_transcript.pdf`` | Transcript of required spanbert query
``gpt3_transcript.pdf`` | Transcript of required gpt3 query
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

A search is launched using the `google-api-python-client`. This returns a JSON object storing each webpage (up to 10 as configured in the Custom search portal) and its corresponding metadata. There is a 20 second timeout for the response of the websites. Only HTML web pages are parsed. If a web page has been visited (visited web pages are maintained with a set) or it has the field `fileFormat` with value `PDF/Adobe Acrobat` in its metadata, it is skipped. Additionally, a response status code of 200 is verified; otherwise, a warning is displayed and the webpage is skipped. To avoid some response codes, a `User-Agent` string is provided in the `headers` argument. We used the following string:
``` python
'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'~
```
BeautifulSoup extracts all the main content with text, and it is processed with regex. Metacharacters representing escape sequences (`\xa0`), tabs (`\t`), new lines (`\n`), are replaced with single spaces. If there is a zero space metacharacter (`\u200b`) it is replaced with an empty string. If the total text is more than 10,000 characters, the remaining text is truncated. 

#### Annotation

Dependencies: `spacy`

The annotation task is done in the `main()` function by making calls to functions in the `spacy_help_functions.py` file, namely `create_entity_pairs()`. The project initializes a `spacy.Doc` object using the pretrained model `en_core_web_lg`. The model tokenizes the text into sentences. For each sentence, the model identifies the existing entities, and the function `create_entity_pairs()` returns the list of entity pairs within a span of the sentence. For each entity pair, if the 2 respective entities match the same entity types for user-provided relation at program launch, then it is collected to be passed to the pretrained language model, SpanBERT. 

#### Extraction

Dependencies: `spacy`, `pytorch-pretrained-bert`, `openai`, `ast`

SpanBERT

Given the entity pairs matching the relation of interest, SpanBERT makes predictions on the relationship between the entities in the pair given the span of context tokens, the subject entity (a tuple of the entity original string, label, and token span), and the object entity. The driving logic for relation extraction is in `extract_relations()`. The original helper function is modified in a few ways: (1) only examples with matching entity type and order of the relation of interest are added to `examples` to predict with SpanBERT (2) if no examples are found (i.e., valid entity pairs) then the sentence is skipped (3) if the predicted relation is not a relation of interest, it is skipped. At the beginning of each sentence, a copy of the values in the result (storing the extracted relations for a given web page document) is made and used to compare with the result values at the end of a sentence processing to determine the number of sentences used. 

GPT-3

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

### Relations

Dependencies: `heapq`

The extracted relations are stored in a RelationSet object, which is a class designed to handle priority ordering efficiently and duplicate elements. The main attributes it stores is a priority queue (implemented with a list and heap), a set, the spaCy relation of interest, and a string corresponding to the specified method of relation extraction at user-input. The following system-defined methods are overwritten for the object

`__len__()`: Returns the length of the queue
`__str__()`: Prints all relations and their confidence (if SpanBERT is specified), subject, and object in the specified format by the reference implementation
`__getitem__()`: Returns the i-th element in sorted order

The `add(element, priority)` method is used to add new relations and returns the number of duplicate relations encountered. Firstly, the provided element (or relation) is checked for membership in the set to handle duplicates. If it is not a duplicate relation, the priority queue pushes the element with priority as the confidence of the relation. The priority queue is maintained as a linear representation of a max heap. The priority of the relation is irrelevant to the case of GPT-3, since all elements have the same confidence/priority. If a given relation is a duplicate, the current priority is compared with the new priority. If the current priority is higher, then no change is made to the RelationSet and the number of duplicate elements increments. If the current priority is lower than the new priority, only if SpanBERT was specified, then the old relation confidence is updated (GPT-3 assumes confidence 1.0 for all relations). The number of duplicate relations is used to compute the number of extracted relations for a given web page, which is the length of the RelationSet object minus the number of duplicates encountered.

The `__getitem()__` system-defined, or dunder, method implementation is particularly used in `update_query()`. It enforces ordered indexing of values by decreasing order of confidence. This simplifies the query update process as the RelationSet object can simply be indexed in order. 


## External references

[1] Query optimizers: http://www.cs.columbia.edu/~gravano/Papers/2008/sigmod-record08.pdf

[2] Spacy documentation: https://spacy.io/api/span
