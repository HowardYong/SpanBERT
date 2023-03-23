from spanbert import SpanBERT
import spacy
from spacy_help_functions import extract_doc_relations_spanbert

raw_text = "Bill Gates stepped down as chairman of Microsoft in February 2014 and assumed a new post as technology adviser to support the newly appointed CEO Satya Nadella."
entities_of_interest = ["ORGANIZATION", "PERSON",
                        "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]

# Load spacy model
nlp = spacy.load("en_core_web_lg")

# Apply spacy model to raw text (to split to sentences, tokenize, extract entities etc.)
doc = nlp(raw_text)

# Load pre-trained SpanBERT model
spanbert = SpanBERT("./pretrained_spanbert")

# Extract relations
relations = extract_doc_relations_spanbert(doc, spanbert, entities_of_interest)
print("Relations: {}".format(dict(relations)))
