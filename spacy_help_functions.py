import ssl
import spacy
from collections import defaultdict
import openai
import re
import ast
import time

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nlp = spacy.load("en_core_web_lg")
SPACY2BERT = {
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE"
}
BERT2SPACY = {
    "ORGANIZATION": "ORG",
    "PERSON": "PERSON",
    "LOCATION": "LOC",
    "CITY": "GPE",
    "COUNTRY": "GPE",
    "STATE_OR_PROVINCE": "GPE",
    "DATE": "DATE"
}
ENTITIES_OF_INTEREST = ["ORGANIZATION", "PERSON",
                        "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
RELATIONS_OF_INTEREST_INTERNAL = {
    1: ['per:schools_attended'],
    2: ['per:employee_of'],
    3: ['per:countries_of_residence', 'per:cities_of_residence', 'per:stateorprovinces_of_residence'],
    4: ['org:top_members/employees']
}
RELATIONS_OF_INTEREST = {
    1: 'Schools_Attended',
    2: 'Work_For',
    3: 'Live_In',
    4: 'Top_Member_Employees'
}
RELATIONS_TO_ENTITIES = {
    1: [ENTITIES_OF_INTEREST[1], ENTITIES_OF_INTEREST[0]],
    2: [ENTITIES_OF_INTEREST[1], ENTITIES_OF_INTEREST[0]],
    3: ENTITIES_OF_INTEREST[1:],
    4: [ENTITIES_OF_INTEREST[0], ENTITIES_OF_INTEREST[1]],
}
RELATION_EXAMPLES = {
    1: '["Jeff Bezos", "Schools_Attended", "Princeton University"]',
    2: '["Alec Radford", "Work_For", "OpenAI"]',
    3: '["Mariah Carey", "Live_In", "New York City"]',
    4: '["Nvidia", "Top_Member_Employees", "Jensen Huang"]'
}


def get_entities(sentence, entities_of_interest):
    return [(e.text, SPACY2BERT[e.label_]) for e in sentence.ents if e.label_ in SPACY2BERT]


def extract_relations(doc, spanbert, r=None, conf=0.7):
    res = defaultdict(int)
    entities_of_interest = RELATIONS_TO_ENTITIES[r]
    relation_of_interest = RELATIONS_OF_INTEREST_INTERNAL[r]

    num_sentences = len([s for s in doc.sents])
    num_sentences_used = 0
    overall_num_relations = 0

    print("\tExtracted {} sentences. Processing each sentence to identify presence of entities of interest...".format(num_sentences))
    c = 0
    for sentence in doc.sents:
        # Copy values in result prior to current sentence to verify if sentence was used.
        old_res = res.copy()
        c += 1
        if c % 5 == 0:
            print(f"\tProcessed {c} / {num_sentences} sentences ")

        # Annotate entities and create pairs using spaCy library.
        entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        examples = []
        for ep in entity_pairs:
            # Add entity pair to extract relation with SpanBERT only if subject and object entity types match
            # the relation of interest.
            if ep[1][1] in entities_of_interest and ep[2][1] in entities_of_interest and ep[1][1] != ep[2][1]:
                subj_ent = tuple(
                    filter(lambda e: e[1] in entities_of_interest[:1], ep))[0]
                obj_ent = tuple(
                    filter(lambda e: e[1] in entities_of_interest[1:], ep))[0]
                examples.append(
                    {"tokens": ep[0], "subj": subj_ent, "obj": obj_ent})
        # Skip this sentence if no entities were labeled.
        if len(examples) == 0:
            continue

        preds = spanbert.predict(examples)
        for ex, pred in list(zip(examples, preds)):
            # Skip the entity pair if no relation found or relation identified is different from relation
            # of interest.
            relation = pred[0]
            if relation == 'no_relation' or relation not in relation_of_interest:
                continue
            overall_num_relations += 1
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
                    print("\t\tAdding to set of extracted relations.")
                else:
                    print(
                        "\t\tDuplicate with lower confidence than existing record. Ignoring this.")
            else:
                print(
                    "\t\tConfidence is lower than threshold confidence. Ignoring this.")
            print("\t\t==========")
        # If results changed after passing through sentence, the sentence was used.
        if len(res.values()) > 0 and res != old_res:
            num_sentences_used += 1
    return res, num_sentences_used, overall_num_relations


def create_entity_pairs(sents_doc, entities_of_interest, window_size=40):
    '''
    Input: a spaCy Sentence object and a list of entities of interest
    Output: list of extracted entity pairs: (text, entity1, entity2)
    '''
    if entities_of_interest is not None:
        entities_of_interest = {BERT2SPACY[b] for b in entities_of_interest}
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
                # Find start of sentence
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
                    e1.text, SPACY2BERT[e1.label_], (e1.start - gap, e1.end - gap - 1))
                e2_info = (
                    e2.text, SPACY2BERT[e2.label_], (e2.start - gap, e2.end - gap - 1))
                if e1.start == e1.end:
                    assert x[e1.start -
                             gap] == e1.text, "{}, {}".format(e1_info, x)
                if e2.start == e2.end:
                    assert x[e2.start -
                             gap] == e2.text, "{}, {}".format(e2_info, x)
                entity_pairs.append((x, e1_info, e2_info))
    return entity_pairs


def extract_relations_gpt3(doc, openai_api_key, r=None, conf=0.7):
    res = defaultdict(int)
    entities_of_interest = RELATIONS_TO_ENTITIES[r]
    relation_of_interest = RELATIONS_OF_INTEREST[r]
    num_sentences = len([s for s in doc.sents])
    num_sentences_used = 0
    overall_num_relations = 0

    # Print a status message to indicate the start of processing.
    print("\tExtracted {} sentences. Processing each sentence to identify presence of entities of interest...".format(num_sentences))

    # Loop through each sentence in the document.
    c = 0
    for sentence in doc.sents:
        # Copy the current state of the result defaultdict.
        old_res = res.copy()
        c += 1

        # Print a status message after every 5 sentences processed.
        if c % 5 == 0:
            print(f"\tProcessed {c} / {num_sentences} sentences ")

        # Create entity pairs from the current sentence.
        entity_pairs = create_entity_pairs(sentence, entities_of_interest)

        # Prepare examples for relation extraction.
        examples = []
        for ep in entity_pairs:
            if ep[1][1] in entities_of_interest and ep[2][1] in entities_of_interest and ep[1][1] != ep[2][1]:
                subj_ent = tuple(
                    filter(lambda e: e[1] in entities_of_interest[:1], ep))[0]
                obj_ent = tuple(
                    filter(lambda e: e[1] in entities_of_interest[1:], ep))[0]
                examples.append(
                    {"tokens": ep[0], "subj": subj_ent, "obj": obj_ent})

        # If there are no examples, skip the current sentence.
        if len(examples) == 0:
            continue

        # Use GPT-3 to extract relations from the current sentence.
        response = extract_relations_sentence_gpt3(
            sentence, entities_of_interest, relation_of_interest, r, openai_api_key)

        # Process the GPT-3 response and extract the relations.
        extracted_relations_list = None
        extracted_relations_string = '[' + \
            re.sub(r'[\n.]', '', response.choices[0].text.strip()) + ']'
        try:
            extracted_relations_list = ast.literal_eval(
                extracted_relations_string)
        except (SyntaxError, AssertionError, ValueError):
            continue

        # Loop through the extracted relations and add them to the result defaultdict.
        gpt3_confidence = 1.00
        for extracted_relation in extracted_relations_list:
            if len(extracted_relation) == 3 and extracted_relation[0] and extracted_relation[1] == relation_of_interest and extracted_relation[2]:
                overall_num_relations += 1
                subj, rel, obj = extracted_relation
                print("\t\t=== Extracted Relation ===")
                print("\t\tSentence: {}".format(sentence))
                print("\t\tSubject: {}\t\tObject: {}".format(subj, obj))

                # If the relation is a duplicate, ignore it.
                if (subj, rel, obj) in res:
                    print("\t\tDuplicate. Ignoring this.")
                # Otherwise, add the relation to the result defaultdict.
                else:
                    print("\t\tAdding to set of extracted relations.")
                    res[(subj, rel, obj)] = gpt3_confidence
                print("\t\t==========")

        # If any relations were added to the result defaultdict, increment the number of sentences used counter.
        if len(res.values()) != 0 and old_res != res:
            num_sentences_used += 1
    return res, num_sentences_used, overall_num_relations


def extract_relations_sentence_gpt3(sentence, entities_of_interest, relation_of_interest, r, openai_api_key):
    # Set OpenAI API key
    openai.api_key = openai_api_key

    # Define the subject and object classifications
    subj_classification = entities_of_interest[0]
    obj_classification = ' or '.join(
        entities_of_interest[1:]) if r == 3 else entities_of_interest[1]

    # Define the prompt for the GPT-3 model
    prompt = (f"Please extract all the {relation_of_interest} relations from the sentence: '{sentence}'. "
              f"Output Format: [\"{subj_classification}\", \"{relation_of_interest}\", \"{obj_classification}\"]  "
              f"Output Example: {RELATION_EXAMPLES[r]}")

    # Set GPT-3 model parameters
    model = 'text-davinci-003'
    max_tokens = 100
    temperature = 0.2
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0

    try:
        # Use the OpenAI API to generate the response
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
    except openai.error.RateLimitError:
        # Catch RateLimitError and sleep for 10 seconds before trying again
        print('Warning: RateLimitError. Sleeping...')
        time.sleep(10)
        extract_relations_sentence_gpt3(
            sentence, entities_of_interest, relation_of_interest, r, openai_api_key)

    return response
