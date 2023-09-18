#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sqlite3
import time
from datetime import datetime
from logging import warn

import bibtexparser
import requests
from bs4 import BeautifulSoup, NavigableString, Tag

USER_AGENT_HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"
}
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EUTILS_SUMMARY_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
CROSSREF_BASE_URL = "https://api.crossref.org/v1/works"
PMC_BASE_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles"
DATABASE_NAME = "articles.db"


def date_str_to_datetime(date_str):
    try:
        return datetime.strptime(date_str, "%Y %b %d").date()
    except ValueError:
        return None


def fetch_from_url(url, params=None):
    try:
        response = requests.get(url, headers=USER_AGENT_HEADER, params=params)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        return response.content
    except requests.RequestException as e:
        print(f"Error: Unable to fetch data from {url}. Reason: {e}")
        return None


def get_article_date(pmc_id):
    params = {
        "db": "pmc",
        "id": pmc_id,
        "tool": "FactYou",
        "email": "seanlaidlaw95@gmail.com",
    }
    response_xml = fetch_from_url(EUTILS_SUMMARY_BASE_URL, params)
    if response_xml:
        soup = BeautifulSoup(response_xml, "xml")
        date_str = (
            soup.find("Item", {"Name": "PubDate"}).text
            if soup.find("Item", {"Name": "PubDate"})
            else ""
        )
        return date_str_to_datetime(date_str)
    return None


def sort_pmc_ids_by_date(pmc_ids):
    pmc_date_pairs = [(pmc_id, get_article_date(pmc_id)) for pmc_id in pmc_ids]
    sorted_pairs = sorted(pmc_date_pairs, key=lambda x: x[1])
    sorted_ids = [pair[0] for pair in sorted_pairs if pair[1] is not None]
    return sorted_ids


def get_pmc_from_doi(doi):
    params = {
        "db": "pmc",
        "term": doi,
        "tool": "FactYou",
        "email": "seanlaidlaw95@gmail.com",
    }
    response_xml = fetch_from_url(EUTILS_BASE_URL, params)
    if response_xml:
        soup = BeautifulSoup(response_xml, "xml")
        error_list = soup.find("ErrorList")
        if not error_list:
            id_list = soup.find("IdList")
            if id_list:
                pmc_ids = [tag.text for tag in id_list.find_all("Id")]
                sorted_pmc_ids = sort_pmc_ids_by_date(pmc_ids)
                oldest_pmc_id = sorted_pmc_ids[0] if sorted_pmc_ids else None
                return oldest_pmc_id
    return None


def clean_up_text(text):
    # if passed text exists and is a tag we need to get text from the tag
    if text:
        if isinstance(text, Tag):
            text = text.text

    # if passed text exists and is a string we need to strip it
    if text:
        text = text.strip()
        text = re.sub(r"<[^>]+>", "", text)  # Removing any HTML tags
        text = re.sub(
            r"^[^A-Za-z]+", "", text
        )  # Stripping everything until the first letter/number/full stop
        text = re.sub(
            r"[^A-Za-z0-9.]$", "", text
        )  # Stripping everything after the last letter/number/full stop

        # remove unicode weirdness that sometimes ends up in strings
        text = re.sub(r"\xa0", " ", text)  # no breaking space
        text = text.strip()
    return text


def fetch_and_parse_article(doi, pmc_id):
    """
    Fetch and parse the article based on given DOI and PMC ID.

    Args:
        doi (str): The DOI of the article.
        pmc_id (str): The PMC ID of the article.

    Returns:
        list: A list of content (sentence/reference pairs) or None.
    """
    if not valid_input(doi, pmc_id):
        warn(
            f"Invalid inputs provided to obtain article from doi ({doi}) / pmcid ({pmc_id})"
        )
        return None

    soup_content = fetch_from_url(f"{PMC_BASE_URL}/PMC{pmc_id}")
    if not soup_content:
        warn(f"Error fetching request to obtain article from pmcid ({pmc_id})")
        return None

    soup = BeautifulSoup(soup_content, "html.parser")
    if not soup:
        warn(f"Error parsing html of article from pmcid ({pmc_id})")

    section_div = find_section_div(soup)
    if not section_div:
        warn(f"Could not find appropriate section for PMC ({pmc_id}) / doi ({doi})")
        return None

    paper_refs = get_references_from_doi(doi)
    if not paper_refs:
        warn(f"Could not find references for doi ({doi})")
        return None
    content_list = build_content_list(section_div, paper_refs, doi)

    return content_list


def valid_input(doi, pmc_id):
    """
    Validate the inputs.

    Args:
        doi (str): The DOI of the article.
        pmc_id (str): The PMC ID of the article.

    Returns:
        bool: True if inputs are valid, False otherwise.
    """
    if not doi:
        warn(f"No valid doi was passed to function. Received doi of value: {doi}")
        return False

    if not pmc_id:
        warn(
            f"No valid PMC ID was passed to function. Received PMC ID of value: {pmc_id}"
        )
        return False

    return True


def build_content_list(section_div, paper_refs, doi):
    """
    Build the content list based on section_div and references.

    Args:
        section_div (bs4.Tag): The section of the article (e.g., Introduction).
        paper_refs (dict): The references of the paper.
        doi (str): The DOI of the article.

    Returns:
        list: A list of content (sentence/reference pairs).
    """
    paragraphs = section_div.find_all("p")
    content_list = []

    for paragraph in paragraphs:
        sentence_fragments, references = extract_text_and_refs(paragraph)
        for text, ref in zip(sentence_fragments, references):
            ref_doi = get_doi_from_reference_number(paper_refs, ref)

            if ref_doi:
                content_list.append(
                    {
                        "Text": text,
                        "Reference": ref,
                        "SrcDOI": doi,
                        "RefDOI": ref_doi,
                        "RefOther": None,
                    }
                )
            else:
                ref_unstruct = get_unstructured_from_reference_number(paper_refs, ref)
                if ref_unstruct:
                    content_list.append(
                        {
                            "Text": text,
                            "Reference": ref,
                            "SrcDOI": doi,
                            "RefDOI": None,
                            "RefOther": ref_unstruct,
                        }
                    )
                else:
                    warn(
                        f"Error: no DOI or unstructured found for Reference: {ref}, in paper: {doi}"
                    )

    return content_list


def extract_text_and_refs(paragraph):
    """
    Extract text and references from the given paragraph.

    Args:
        paragraph (bs4.Tag): A paragraph from the article.

    Returns:
        tuple: A tuple containing lists of sentence fragments and references.
    """

    # collapse structure of paragraph to a list of strings and <a> tags
    paragraph_elements = []
    for child in paragraph.children:
        if isinstance(child, str):
            paragraph_elements.append(child)
        if isinstance(child, Tag):
            if child.name == "a" and "bibr" in child.get("class", []):
                paragraph_elements.append(child)
            elif child.name == "sup":
                for link in child.find_all("a", {"class": "bibr"}):
                    paragraph_elements.append(link)

    texts, references = [], []
    current_text, just_processed_ref = "", False

    for child in paragraph_elements:
        # we check to see if the child is or contains a reference as we accumulate
        # strings until we reach a reference at which we save what we have as the
        # reference's corresponding text
        if contains_reference(child):
            current_ref = extract_reference(child)
            references.append(current_ref)
            # if there is no current text and we just processed a reference, it means this reference is in relation to last text
            if not current_text and just_processed_ref and texts:
                texts.append(texts[-1])
            else:
                current_text = current_text.strip()
                if current_text:
                    texts.append(current_text)
                just_processed_ref = True
                current_text = ""
            continue
        child = clean_up_text(child)
        if child:
            just_processed_ref = False
            current_text += child

    if len(texts) != len(references):
        warn(
            f"Error: number of sentences ({len(texts)}) and references ({len(references)}) do not match for paragraph: {paragraph}"
        )
    return texts, references


# def merge_style_tags(children):
# merged = []
# for child in children:
# if isinstance(child, NavigableString):
# # For text nodes, sup, or a tags, just append to the list
# merged.append(child)
# elif child.name in ["em", "strong", "i"]:
# # For style tags, add content to the previous node if it's a text node
# # Else append as a new node
# if merged and isinstance(merged[-1], NavigableString):
# merged[-1] = NavigableString(merged[-1] + child.string)
# else:
# merged.append(child.string)
# return merged


def contains_bibr_reference(tag):
    return tag.find("a", class_="bibr") is not None


def merge_style_tags(children):
    merged = []
    buffer = ""

    for child in children:
        if isinstance(child, NavigableString):
            buffer += child
        elif contains_bibr_reference(child):
            # For tags containing <a class="bibr">, append the buffer and the reference to the list
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            # You can either add the entire child or just the nested <a class="bibr">
            # merged.append(child.find("a", class_="bibr"))
            merged.append(child)
        elif child.name in ["em", "strong", "i"]:
            # For style tags, add content to the buffer
            buffer += child.get_text()
        else:
            buffer += child.get_text()

    # Append any remaining buffer to the merged list
    if buffer:
        merged.append(buffer.strip())
    return merged


def contains_reference(child):
    """
    Check if the child contains a reference.

    Args:
        child (bs4.Tag or bs4.NavigableString): A child element of the paragraph.

    Returns:
        bool: True if it contains a reference, False otherwise.
    """
    if isinstance(child, Tag):
        condition1 = child.name == "a" and "bibr" in child.get("class", [])
        condition2 = child.find_all("a", attrs={"class": "bibr"})
        return condition1 or bool(condition2)

    return False


def extract_reference(tag):
    """
    Extract the reference number from the tag.

    Args:
        tag (bs4.Tag): The tag containing the reference.

    Returns:
        str: The extracted reference number or an empty string if not found.
    """
    if tag.name != "a":
        tag = tag.find("a", {"class": "bibr"})

    rid = tag.attrs.get("rid", "")
    match = re.search(r"(\d+)", rid)
    if match:
        return match.group(1)
    else:
        warn(f"No number found for sentence: {tag}")
        return ""


def get_references_from_doi(doi):
    data = fetch_from_url(f"{CROSSREF_BASE_URL}/{doi}")
    if data:
        try:
            json_data = json.loads(data)
            return json_data["message"].get("reference", [])
        except json.JSONDecodeError:
            print(f"Error decoding JSON data for DOI: {doi}")
    return []


def get_doi_from_reference_number(references, ref_number):
    if ref_number == None:
        print("Ref number is None")
        return None
    if ref_number is not None:
        try:
            index = int(ref_number) - 1
            try:
                # Now use index to access references
                return references[index].get("DOI", None)
            except IndexError:
                print(f"Reference number {ref_number} not found")
                return None
        except ValueError:
            print("DOI not found")
            return None


def get_unstructured_from_reference_number(references, ref_number):
    if ref_number is not None:
        try:
            index = int(ref_number) - 1
            try:
                # Now use index to access references
                ref = references[index]
                if "unstructured" in ref:
                    return ref.get("unstructured")
                elif len(ref.keys()) > 0:
                    return json.dumps(ref)
                else:
                    warn(
                        f"Can't find any info for reference number {ref_number} in paper"
                    )
                    return None
            except IndexError:
                warn(f"Reference number {ref_number} not found")
                return None
        except ValueError:
            print("DOI not found")
            return None


def find_section_div(soup):
    div_ids = ["S1", "Sec1", "s1", "sec001", "s001"]
    for div_id in div_ids:
        div = soup.find("div", id=div_id)
        if div:
            return div

    h2_tags = soup.find_all("h2")

    for h2 in h2_tags:
        if (
            "introduction" in h2.text.lower()
            or "background" in h2.text.lower()
            or h2.text.lower() == "main"
        ):
            if h2.parent and h2.parent.name == "div":
                section_div = h2.parent
                return section_div
    else:
        # if <h2> tag is not found
        return None


def extract_dois_from_bib(bib_file_path):
    with open(bib_file_path, "r") as bibfile:
        bib_database = bibtexparser.load(bibfile)
    return [
        entry.get("doi", "").strip() for entry in bib_database.entries if "doi" in entry
    ]


def get_all_dois_from_folder(bib_folder):
    all_dois = []
    for bib_file in os.listdir(bib_folder):
        if bib_file.endswith(".bib") and bib_file != "paperpile.bib":
            bib_file_path = os.path.join(bib_folder, bib_file)
            all_dois.extend(extract_dois_from_bib(bib_file_path))
    return all_dois


class ArticleDatabase:
    def __init__(self, db_name="articles.db"):
        self.conn = sqlite3.connect(db_name)
        self._setup_tables()

    def _setup_tables(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS Referenced (
                    Text TEXT,
                    TextWtContext TEXT,
                    TextEmbeddings BLOB,
                    Reference INTEGER,
                    SrcDOI TEXT,
                    RefDOI TEXT,
                    RefOther TEXT
                )
            """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS Scanned (
                    DOI TEXT,
                    PMCID TEXT,
                    Parsed BOOLEAN NOT NULL CHECK (Parsed IN (0, 1)),
                    Hashed BOOLEAN NOT NULL CHECK (Hashed IN (0, 1)),
                    Skipped BOOLEAN NOT NULL CHECK (Skipped IN (0, 1))
                )
            """
            )

    def store_articles(self, content_list):
        with self.conn:
            self.conn.executemany(
                "INSERT INTO Referenced (Text, Reference, SrcDOI, RefDOI, RefOther) VALUES (:Text, :Reference, :SrcDOI, :RefDOI, :RefOther)",
                content_list,
            )

    def save_scanned_doi(self, doi, pmc_id, skipped=0):
        with self.conn:
            self.conn.execute(
                "INSERT INTO Scanned VALUES (:DOI, :PMCID, :Parsed, :Hashed, :Skipped)",
                (doi, pmc_id, 0, 0, skipped),
            )

    def update_scanned_doi(self, doi, parsed=None, hashed=None):
        if parsed:
            with self.conn:
                self.conn.execute(
                    "UPDATE Scanned SET Parsed = 1 WHERE DOI = ?;", (doi,)
                )
        if hashed:
            with self.conn:
                self.conn.execute(
                    "UPDATE Scanned SET Hashed = 1 WHERE DOI = ?;", (doi,)
                )

    def doi_exists(self, doi):
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM Scanned WHERE DOI = ?", (doi,))
        return cursor.fetchone() is not None

    def get_pmc_articles(self):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT DOI, PMCID FROM Scanned WHERE Skipped == 0 AND Parsed == 0;"
        )
        return cursor.fetchall()

    def close(self):
        self.conn.close()


def main():
    bib_folder = "/Users/sl31/My Drive/Paperpile/Bib Exports"
    bib_dois = get_all_dois_from_folder(bib_folder)

    db = ArticleDatabase()

    for doi in bib_dois:
        if not db.doi_exists(doi):
            pmc_id = get_pmc_from_doi(doi)
            if pmc_id:
                db.save_scanned_doi(doi, pmc_id)
            else:
                db.save_scanned_doi(doi=doi, pmc_id=None, skipped=1)
                print(f"No PMC version of {doi} was available")

    for doi, pmc_id in db.get_pmc_articles():
        print(f"Running on paper: (doi: {doi}) (pmcid: {pmc_id})")
        content_list = fetch_and_parse_article(doi=doi, pmc_id=pmc_id)
        if content_list:
            db.store_articles(content_list)
            db.update_scanned_doi(parsed=1, doi=doi)
        else:
            warn(f"Error obtaining data for doi: {doi}")
        time.sleep(5)

    db.close()


if __name__ == "__main__":
    main()
