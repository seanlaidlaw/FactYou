#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sqlite3

import bibtexparser
import requests
from bs4 import BeautifulSoup

# Set user-agent headers
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
}


def get_pmc_from_doi(doi):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",
        "term": doi,
        "tool": "FactYou",
        "email": "seanlaidlaw95@gmail.com",
    }

    response = requests.get(base_url, params=params)
    response_xml = response.content

    pmc_id = None

    # Parsing XML response to extract the PMC ID
    soup = BeautifulSoup(response_xml, "xml")
    id_list = soup.find("IdList")
    if id_list:
        pmc_id_tag = id_list.find("Id")
        if pmc_id_tag:
            pmc_id = pmc_id_tag.text

    return pmc_id


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
                sentence_fragments.append(clean_up_text(current_text.strip()))

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


def clean_up_text(text):
    # Remove all HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove all characters before the first [A-Za-z] appears
    text = re.sub(r"^[^A-Za-z]+", "", text)

    return text


def get_references_from_doi(doi):
    """Fetch the reference array for a given DOI."""
    url = f"https://api.crossref.org/v1/works/{doi}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error {response.status_code}: Unable to fetch data for DOI {doi}")
        return []

    data = response.json()
    references = data["message"].get("reference", [])
    return references


def get_doi_from_reference_number(references, ref_number):
    """Retrieve the DOI from a specific reference entry by its number."""
    ref_number = int(ref_number)
    try:
        ref_entry = references[ref_number - 1]  # Adjusting for 0-based indexing
        return ref_entry.get("DOI", "DOI not found")
    except IndexError:
        print(f"Reference number {ref_number} not found in DOI {doi}")
        return None


def setup_database():
    conn = sqlite3.connect("articles.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS Referenced (
            Text TEXT,
            Reference INTEGER,
            SrcDOI TEXT,
            RefDOI TEXT
        )
    """
    )
    conn.commit()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS Scanned (
            DOI TEXT
        )
    """
    )
    conn.commit()
    return conn


def store_to_database(conn, content_list):
    c = conn.cursor()
    c.executemany(
        "INSERT INTO Referenced VALUES (:Text, :Reference, :SrcDOI, :RefDOI)",
        content_list,
    )
    conn.commit()


def save_src_to_database(conn, doi):
    c = conn.cursor()
    c.execute("INSERT INTO Scanned VALUES (:DOI)", [str(doi)])
    conn.commit()


def fetch_and_parse_article(doi):
    pmc_id = get_pmc_from_doi(doi)
    if not pmc_id:
        print(f"No PMC version of {doi} was available")
        return

    pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}"
    response = requests.get(pmc_url, headers=headers)

    if response.status_code != 200:
        print(f"Error fetching content for {doi}")
        return

    soup = BeautifulSoup(response.content, "html.parser")
    section_div = find_section_div(soup)

    if not section_div:
        print(f"Could not find appropriate section for PMC ({pmc_id}) / doi ({doi})")
        return

    paper_refs = get_references_from_doi(doi)
    paragraphs = section_div.find_all("p")

    return build_content_list(paragraphs, paper_refs, doi)


def find_section_div(soup):
    # List of potential div IDs to check
    div_ids = ["S1", "Sec1", "s1", "sec001", "s001"]

    for div_id in div_ids:
        div = soup.find("div", id=div_id)
        if div:
            return div

    # If none of the above IDs match, look for the h2 containing Introduction/Background
    h2_tag = soup.find(
        "h2",
        string=lambda s: ("introduction" in s.lower()) or ("background" in s.lower()),
    )
    return h2_tag.find_parent("div") if h2_tag else None


def build_content_list(paragraphs, paper_refs, doi):
    content_list = []
    for paragraph in paragraphs:
        sentence_fragments, references = extract_text_and_refs(paragraph)
        for text, ref in zip(sentence_fragments, references):
            ref_doi = get_doi_from_reference_number(paper_refs, ref)
            if ref_doi and ref_doi != "DOI not found":
                content_list.append(
                    {"Text": text, "Reference": ref, "SrcDOI": doi, "RefDOI": ref_doi}
                )
            else:
                print(f"Error: no DOI found for Reference: {ref}, in paper: {doi}")

    if not content_list:
        print(f"Couldn't find any sentence/references for PMC article with DOI {doi}")
        return
    return content_list


# def fetch_and_parse_article(doi):
# pmc_id = get_pmc_from_doi(doi)
# if not pmc_id:
# print(f"No PMC version of {doi} was available")
# return None

# pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}"
# response = requests.get(pmc_url, headers=headers)
# soup = BeautifulSoup(response.content, 'html.parser')


# section_div = soup.find('div', id='S1')
# if not section_div:
# section_div = soup.find('div', id='Sec1')
# if not section_div:
# section_div = soup.find('div', id='s1')
# if not section_div:
# section_div = soup.find('div', id='sec001')
# if not section_div:
# section_div = soup.find('div', id='s001')
# # or look for a h2 with text containing Introduction or Background
# if not section_div:
# # Find the first <h2> that contains "introduction" or "background" (case insensitive)
# h2_tag = soup.find('h2', string=lambda s: ("introduction" in s.lower()) or ("background" in s.lower()))

# if not h2_tag:
# print("No matching <h2> tag found!")
# return None
# # Find the containing div for that <h2>
# section_div = h2_tag.find_parent('div')
# if not section_div:
# print(f"Div with id 'S1' or 'Sec1' not found! for PMC ({pmc_id}) / doi ({doi})")
# return None

# paper_refs = get_references_from_doi(doi)
# paragraphs = section_div.find_all('p')

# content_list = []
# for paragraph in paragraphs:
# sentence_fragments, references = extract_text_and_refs(paragraph)
# for text, ref in zip(sentence_fragments, references):
# ref_doi = get_doi_from_reference_number(paper_refs, ref)
# if ref_doi:
# if ref_doi != "DOI not found":
# content_list.append({'Text': text, 'Reference': ref, 'SrcDOI': doi, 'RefDOI': ref_doi})
# else:
# print(f"Error: no DOI found for Reference: {ref}, in paper: {doi}")

# if len(content_list) == 0:
# print(f"Couldn't find any sentence / references for PMC article: {pmc_id}")
# return None
# return content_list


def extract_dois_from_bib(bib_file_path):
    """Extract DOIs from a .bib file."""
    with open(bib_file_path, "r") as bibfile:
        bib_database = bibtexparser.load(bibfile)

    # Extract DOIs from entries
    return [
        entry.get("doi", "").strip() for entry in bib_database.entries if "doi" in entry
    ]


def get_all_dois_from_folder(bib_folder):
    """Get all DOIs from all .bib files in the specified folder."""
    all_dois = []

    # Loop over all files in the folder
    for bib_file in os.listdir(bib_folder):
        # only look at bib files
        if bib_file.endswith(".bib"):
            # exclude my export of everything
            if bib_file != "paperpile.bib":
                bib_file_path = os.path.join(bib_folder, bib_file)
                all_dois.extend(extract_dois_from_bib(bib_file_path))

    return all_dois


def doi_exists_in_src_db(conn, doi):
    """Check if a DOI already exists in the SrcDOI column of the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM Referenced WHERE SrcDOI = ?", (doi,))
    result = cursor.fetchone()
    return result is not None


if __name__ == "__main__":
    bib_folder = "/Users/seanlaidlaw/Library/CloudStorage/GoogleDrive-seanlaidlaw95@gmail.com/.shortcut-targets-by-id/16zz-HLLhNYcHVuI8ONmvn-ed_kEi7Hnn/Paperpile/Bib Exports/"
    all_dois = get_all_dois_from_folder(bib_folder)

    # Establish a connection to the database
    # conn = setup_database()

    for doi in all_dois:
        print(f"Running on doi: {doi}")
        # if not doi_exists_in_src_db(conn, doi):
        content_list = fetch_and_parse_article(doi)
        # if content_list:
        # store_to_database(conn, content_list)
        # else:
        # print(f"Error obtaining data and working for doi: {doi}")

        # # Wait for 5 seconds before processing the next DOI
        # time.sleep(5)
        # else:
        # print(f"DOI {doi} already exists in the database, skipping...")

        # # save the DOI to database so we don't retry it
        # save_src_to_database(conn, doi)

    # conn.close()
