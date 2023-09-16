#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

import requests
import torch
from bs4 import BeautifulSoup
from scipy.spatial.distance import cosine
from transformers import BertModel, BertTokenizer

# Specify the file name for caching
CACHE_FILE = "webpage_cache.html"
url = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9882184"

# Set user-agent headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
}


def extract_text_and_refs(paragraph):
    """Extract text and references from a paragraph."""
    sentence_fragments = []
    references = []
    current_text = ""

    for child in paragraph.children:
        # If the child is an "a" tag with class "bibr" or contains one
        if isinstance(child, (str, bytes)):
            current_text += child
        elif (
            child.name == "a"
            and "bibr" in child.get("class", [])
            or child.find("a", class_="bibr")
        ):
            ref = child.find("a", class_="bibr") if child.name != "a" else child
            if ref:
                sentence_fragments.append(current_text.strip())

                # Extract only the number from the 'rid' attribute
                rid = ref.attrs.get("rid", "")
                match = re.search(r"(\d+)", rid)
                ref_number = match.group(1) if match else "No number found"

                references.append(ref_number)
                current_text = ""
        else:
            current_text += str(child)

    # To handle any remaining text after the last reference
    if current_text.strip():
        sentence_fragments.append(current_text.strip())

    return sentence_fragments, references


# Check if cache exists
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        content = f.read()
else:
    response = requests.get(url, headers=headers)
    content = response.content

    # Store the HTML content to the cache file
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        f.write(content.decode("utf-8"))

soup = BeautifulSoup(content, "html.parser")


# Find the div with id "S1"
section_div = soup.find("div", id="S1")

if section_div:
    # Extract all the content within the div
    paragraphs = section_div.find_all("p")

    content_list = []
    for paragraph in paragraphs:
        sentence_fragments, references = extract_text_and_refs(paragraph)

        for text, ref in zip(sentence_fragments, references):
            content_list.append((text, ref))

    for text, ref_text in content_list:
        print(f"Text: {text}\nReference: {ref_text}\n\n")
else:
    print("Div with id 'S1' not found!")
    exit()


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


def get_embedding(sentence):
    inputs = tokenizer(
        sentence, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def find_most_similar(new_sentence, table):
    new_embedding = get_embedding(new_sentence)
    min_distance = float("inf")
    most_similar_sentence = None

    for entry in table:
        sentence, _ = entry  # Unpack the tuple into sentence and reference
        sentence_embedding = get_embedding(sentence)
        distance = cosine(new_embedding, sentence_embedding)

        if distance < min_distance:
            min_distance = distance
            most_similar_sentence = sentence

    return most_similar_sentence


# Test
new_sentence = (
    "clinical phenotypes in affected central nervous regions associate with TDP-43"
)
print(find_most_similar(new_sentence, content_list))
