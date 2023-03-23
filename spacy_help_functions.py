import ast
import re
import ssl
import openai
import spacy
from collections import defaultdict

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nlp = spacy.load("en_core_web_lg")
spacy2bert = {
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE"
}
bert2spacy = {
    "ORGANIZATION": "ORG",
    "PERSON": "PERSON",
    "LOCATION": "LOC",
    "CITY": "GPE",
    "COUNTRY": "GPE",
    "STATE_OR_PROVINCE": "GPE",
    "DATE": "DATE"
}


def get_entities(sentence, entities_of_interest):
    return [(e.text, spacy2bert[e.label_]) for e in sentence.ents if e.label_ in spacy2bert]


def extract_doc_relations_spanbert(doc, spanbert, entities_of_interest=None, conf=0.7):
    res = defaultdict(int)
    num_sentences = len([s for s in doc.sents])
    num_sentences_used = 0

    print("\tExtracted {} sentences. Processing each sentence to identify presence of entities of interest...".format(num_sentences))
    c = 0
    for sentence in doc.sents:
        c += 1
        if c % 5 == 0:
            print(f"\tProcessed {c} / {num_sentences} sentences ")

        entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        examples = []
        for ep in entity_pairs:
            if ep[1][1] not in entities_of_interest[:1] or ep[2][1] not in entities_of_interest[1:]:
                continue
            examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
        if len(examples) == 0:
            continue

        preds = spanbert.predict(examples)
        for ex, pred in list(zip(examples, preds)):
            relation = pred[0]
            if relation == 'no_relation':
                continue
            print("\t\t=== Extracted Relation ===")
            print("\t\tTokens: {}".format(ex['tokens']))
            subj = ex["subj"][0]
            obj = ex["obj"][0]
            confidence = pred[1]
            print("\t\tRelation: {} (Confidence: {:.3f})\n\t\tSubject: {}\t\tObject: {}".format(
                relation, confidence, subj, obj))
            if confidence > conf:
                if res[(subj, relation, obj)] < confidence:
                    res[(subj, relation, obj)] = confidence
                    num_sentences_used += 1
                    print("\t\tAdding to set of extracted relations.")
                else:
                    print(
                        "\t\tDuplicate with lower confidence than existing record. Ignoring this.")
            else:
                print(
                    "\t\tConfidence is lower than threshold confidence. Ignoring this.")
            print("\t\t==========")
    return res, num_sentences_used


def extract_doc_relations_gpt3(doc, openai_api_key, relation, entities_of_interest=None):
    res = defaultdict(int)
    num_sentences = len([s for s in doc.sents])
    num_sentences_used = 0

    print("\tExtracted {} sentences. Processing each sentence to identify presence of entities of interest...".format(num_sentences))
    c = 0
    for sentence in doc.sents:
        c += 1
        if c % 5 == 0:
            print(f"\tProcessed {c} / {num_sentences} sentences ")

        entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        for ep in entity_pairs:
            if ep[1][1] not in entities_of_interest[:1] or ep[2][1] not in entities_of_interest[1:]:
                continue
            extracted_relation = extract_sentence_relations_gpt3(
                sentence, openai_api_key, relation)
            if extracted_relation:
                num_sentences_used += 1
                print("extracted relation was valid, adding ", extracted_relation)
                for relation in extracted_relation:
                    res[extracted_relation] = 1.0
            else:
                print("extracted relation was invalid")

    return res, num_sentences_used


def extract_sentence_relations_gpt3(sentence, openai_api_key, relation_type):
    openai.api_key = openai_api_key

    examples = {
        'Schools_Attended': '["Jeff Bezos", "Schools_Attended", "Princeton University"]',
        'Work_For': '["Alec Radford", "Work_For", "OpenAI"]',
        'Live_In': '["Mariah Carey", "Live_In", "New York City"]',
        'Top_Member_Employees': '["Jensen Huang", "Top_Member_Employees", "Nvidia"]'
    }

    entities_of_interest = ["ORGANIZATION", "PERSON",
                            "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
    relation_to_entities = {
        'Schools_Attended': {'Subject': entities_of_interest[1], 'Object': entities_of_interest[0]},
        'Work_For': {'Subject': entities_of_interest[1], 'Object': entities_of_interest[0]},
        'Live_In': {'Subject': entities_of_interest[1], 'Object': entities_of_interest[2:]},
        'Top_Member_Employees': {'Subject': entities_of_interest[0], 'Object': entities_of_interest[1]}
    }

    entity_classification = relation_to_entities[relation_type]['Subject']
    object_classification = relation_to_entities[relation_type]['Object']

    if relation_type == 'Live_In':
        object_classification = " or ".join(object_classification.split())

    prompt = f"Given the following example of a relation: {examples[relation_type]}, please extract all the {relation_type} relations in the format of [\"Entity1\", \"{relation_type}\", \"Entity2\"] from the sentence: '{sentence}'. If the relation is not directly mentioned in the sentence, infer the correct one based on the context and the provided example. List all relevant relations. Entity1 should fall under the classification of {entity_classification} and Entity2 should fall under the classification of {object_classification}."

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

    relations_string = '[' + re.sub(r'[\n.]',
                                    '', response.choices[0].text.strip()) + ']'
    try:
        relations_list = ast.literal_eval(relations_string)
        validated_relations = []
        for relation in relations_list:
            if len(relation) == 3 and relation[0] and relation[1] == relation_type and relation[2]:
                validated_relations.append(relation)
        return validated_relations
    except (SyntaxError, AssertionError, ValueError):
        return None


def create_entity_pairs(sents_doc, entities_of_interest, window_size=40):
    '''
    Input: a spaCy Sentence object and a list of entities of interest
    Output: list of extracted entity pairs: (text, entity1, entity2)
    '''
    if entities_of_interest is not None:
        entities_of_interest = {bert2spacy[b] for b in entities_of_interest}
    ents = sents_doc.ents  # get entities for given sentence

    length_doc = len(sents_doc)
    entity_pairs = []
    for i in range(len(ents)):
        e1 = ents[i]
        if entities_of_interest is not None and e1.label_ not in entities_of_interest:
            continue

        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            if entities_of_interest is not None and e2.label_ not in entities_of_interest:
                continue
            if e1.text.lower() == e2.text.lower():  # make sure e1 != e2
                continue

            if (1 <= (e2.start - e1.end) <= window_size):

                punc_token = False
                start = e1.start - 1 - sents_doc.start
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0

                # Find end of sentence
                punc_token = False
                start = e2.end - sents_doc.start
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc

                if (right_r - left_r) > window_size:  # sentence should not be longer than window_size
                    continue

                x = [token.text for token in sents_doc[left_r:right_r]]
                gap = sents_doc.start + left_r
                e1_info = (
                    e1.text, spacy2bert[e1.label_], (e1.start - gap, e1.end - gap - 1))
                e2_info = (
                    e2.text, spacy2bert[e2.label_], (e2.start - gap, e2.end - gap - 1))
                if e1.start == e1.end:
                    assert x[e1.start -
                             gap] == e1.text, "{}, {}".format(e1_info, x)
                if e2.start == e2.end:
                    assert x[e2.start -
                             gap] == e2.text, "{}, {}".format(e2_info, x)
                entity_pairs.append((x, e1_info, e2_info))
    return entity_pairs
