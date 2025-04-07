import json
import os
import re
import sqlite3
from datetime import datetime
from logging import warn

import bibtexparser
import requests
import spacy
from bs4 import BeautifulSoup, NavigableString, Tag

from factyu.config import (CROSSREF_BASE_URL, DB_PATH, EUTILS_BASE_URL,
                           EUTILS_SUMMARY_BASE_URL, PMC_BASE_URL,
                           USER_AGENT_HEADER)
from factyu.database.models import ArticleDatabase
from factyu.extraction.references import (fetch_html_from_PMCID,
                                          get_pmc_from_doi)


def fetch_and_parse_article(doi, pmc_id):
    print(f"Attempting to parse article DOI: {doi}, PMC: {pmc_id}")

    # standardize and clean up DOI and PMCid
    # this removes URL and 'PMC' prefixes
    doi = clean_doi(doi)
    pmc_id = clean_pmc_id(pmc_id)

    if not valid_input(doi, pmc_id):
        warn(f"Invalid inputs provided for doi ({doi}) / pmcid ({pmc_id})")
        return None

    # Fetch HTML content (will use cache if available)
    html_content = fetch_html_from_PMCID(pmc_id)
    if not html_content:
        warn(f"Error fetching article for pmcid ({pmc_id})")
        return None

    print(f"Successfully fetched content for PMC{pmc_id}")

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")
    if not soup:
        warn(f"Error parsing html for pmcid ({pmc_id})")
        return None

    section_div = find_section_div(soup)
    if not section_div:
        warn(f"Could not find appropriate section for PMC ({pmc_id}) / doi ({doi})")
        return None

    paper_refs = get_references_from_doi(doi)
    if not paper_refs:
        warn(f"Could not find references for doi ({doi})")
        return None
    print(f"Found {len(paper_refs)} references for DOI {doi}")

    content_list = build_content_list(section_div, paper_refs, doi)
    if content_list:
        print(f"Successfully extracted {len(content_list)} sentences with references")
    else:
        print("No sentences with references were extracted")
    return content_list


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


def run_extraction(
    bib_folder, user_email, db_path=None, progress_callback=None, final_callback=None
):
    """
    Run the extraction process with progress updates.
    """
    # 1. Extract all DOIs from the provided folder.
    bib_dois = get_all_dois_from_folder(bib_folder)
    if progress_callback:
        progress_callback(0, f"Found {len(bib_dois)} DOIs in the bibliography folder.")

    # Initialize database only when needed
    from factyu.database.models import ArticleDatabase

    db = ArticleDatabase(db_path) if db_path else ArticleDatabase()

    # Filter out DOIs that are already fully processed (parsed=1)
    # This prevents re-processing articles that are already in the database
    # but still allows adding new DOIs or retrying previously failed ones
    processed_dois = db.get_processed_dois()
    new_dois = [doi for doi in bib_dois if doi not in processed_dois]

    if progress_callback:
        progress_callback(
            5, f"Processing {len(new_dois)} new DOIs out of {len(bib_dois)} total"
        )

    try:
        # 2. Scan the DOIs and save those not yet scanned.
        for doi in new_dois:
            if not db.doi_exists(doi):
                pmc_id = get_pmc_from_doi(doi, user_email)
                if pmc_id:
                    db.save_scanned_doi(doi, pmc_id)
                else:
                    db.save_scanned_doi(doi, None, skipped=1)
                    if progress_callback:
                        progress_callback(
                            10, f"No PMC version available for DOI: {doi}"
                        )

        # 3. Process each article with a valid PMC ID.
        # Only process articles that haven't been successfully parsed yet
        articles = db.get_unparsed_pmc_articles()
        total_articles = len(articles)

        if total_articles == 0:
            if progress_callback:
                progress_callback(100, "No new articles to process.")
            if final_callback:
                final_callback()
            return

        for idx, (doi, pmc_id) in enumerate(articles):
            try:
                if progress_callback:
                    percentage = 10 + int(90 * idx / total_articles)
                    progress_callback(
                        percentage, f"Processing article: DOI: {doi}, PMCID: {pmc_id}"
                    )
                content_list = fetch_and_parse_article(doi, pmc_id)
                if content_list:
                    db.store_articles(content_list)
                    db.update_scanned_doi(doi, parsed=True)
                else:
                    warn_msg = f"Error obtaining data for DOI: {doi}"
                    if progress_callback:
                        progress_callback(10 + int(90 * idx / total_articles), warn_msg)
            except Exception as e:
                warn(f"Error processing DOI {doi}: {str(e)}")

        if progress_callback:
            progress_callback(100, "Extraction complete.")

    finally:
        db.close()

    # Call the final callback if provided
    if final_callback:
        final_callback()


def clean_up_text(text):
    if text:
        if isinstance(text, Tag):
            text = text.text
    if text:
        text = text.strip()
        # Remove HTML tags.
        text = re.sub(r"<[^>]+>", " ", text)
        # Remove non-alphabetic characters at the beginning.
        text = re.sub(r"^[^A-Za-z]+", " ", text)
        # Remove a non-alphanumeric period at the end.
        text = re.sub(r"[^A-Za-z0-9.]$", " ", text)
        text = re.sub(r"\xa0", " ", text)
        text = text.strip()
        text = re.sub(r" +", " ", text)
        # Remove dangling punctuation in brackets:
        # This matches either (...) or [...] where the content is only punctuation (commas, semicolons, periods, hyphens) and whitespace.
        text = re.sub(r"[\(\[]\s*[,;.\-]+\s*[\)\]]", "", text)
        # Remove any empty brackets that may remain, e.g. () or [].
        text = re.sub(r"[\(\[]\s*[\)\]]", "", text)
        # Remove trailing punctuation marks (commas, semicolons, colons, periods, hyphens) and whitespace at the end.
        text = re.sub(r"\s*[,\.;:\-]+\s*$", "", text)
        text = text.strip()
    return text


def clean_doi(doi):
    """Clean and extract the DOI from various URL formats"""
    if not doi:
        return None

    # Remove common DOI URL prefixes
    prefixes = [
        r"^https://api\.crossref\.org/v1/works/",
        r"^api\.crossref\.org/v1/works/",
        r"^https://dx\.doi\.org/",
        r"^dx\.doi\.org/",
    ]

    for prefix in prefixes:
        doi = re.sub(prefix, "", doi)

    return doi


def clean_pmc_id(pmc_id):
    """Clean and extract the PMC ID"""
    if not pmc_id:
        return None

    # Remove PMC prefix if present
    pmc_id = re.sub(r"^PMC", "", pmc_id)

    return pmc_id


def valid_input(doi, pmc_id):
    """Validate DOI and PMC ID inputs"""
    if not doi:
        warn(f"Invalid DOI format: {doi}")
        return False

    # use regex to validate DOI
    # DOI regex pattern from https://www.crossref.org/blog/dois-and-matching-regular-expressions/
    doi_pattern = r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$"

    if not re.match(doi_pattern, doi, re.IGNORECASE):
        return False

    if not pmc_id:
        warn(f"Invalid PMC ID format: {pmc_id}")
        return False

    # PMC ID should be numeric
    if not pmc_id.isdigit():
        return False

    # if it hasnt failed above checks, return true
    return True


def convert_square_bracket_refs(html):
    """
    Replace square-bracketed references with <sup> wrapped references.

    For example:
      [<a ...>1</a>, <a ...>2</a>]  --> <sup><a ...>1</a>, <a ...>2</a></sup>
    """
    # If html is not a string, convert it (this handles BeautifulSoup objects).
    if not isinstance(html, str):
        html = str(html)

    # Pattern to match content inside square brackets.
    pattern = re.compile(r"\[([^\]]+)\]")

    def replacer(match):
        # Get inner content and strip extra whitespace.
        content = match.group(1).strip()
        return f"<sup>{content}</sup>"

    replaced_html = pattern.sub(replacer, html)
    replaced_soup = BeautifulSoup(replaced_html, "html.parser")
    return replaced_soup


def build_content_list(section_div, paper_refs, doi):
    # section_div might contain non-reference information after a given subsection
    # we transverse the div until we find a mention of a figure or a supplementary (<a> with href starting with #S or #F)
    # and then we determine the section that is in and truncate section_div to just before that section starts
    # Find the first figure/supplementary reference
    first_supp_or_fig = section_div.find(
        lambda tag: tag.name == "a" and tag.get("href", "").startswith(("#S", "#F"))
    )

    # If found, get its parent section/div and truncate section_div content
    if first_supp_or_fig:
        parent = first_supp_or_fig
        while parent and parent.name not in ["section", "div"]:
            parent = parent.parent

        if parent:
            # Remove this section and all following siblings
            current = parent
            while current:
                next_sib = current.next_sibling
                current.decompose()
                current = next_sib

    paragraphs = section_div.find_all("p")
    content_list = []
    for paragraph in paragraphs:
        sentences, sentence_fragments, references = extract_text_and_refs(paragraph)
        for context, text, ref in zip(sentences, sentence_fragments, references):
            ref_doi = get_doi_from_reference_number(paper_refs, ref)
            if ref_doi:
                content_list.append(
                    {
                        "Text": text,
                        "Reference": ref,
                        "TextInSentence": context,
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
                            "TextInSentence": context,
                            "SrcDOI": doi,
                            "RefDOI": None,
                            "RefOther": ref_unstruct,
                        }
                    )
                else:
                    warn(
                        f"Error: no DOI or unstructured info found for ref {ref} in paper {doi}"
                    )
    return content_list


def spacy_sentence_html_mapping(html_str):
    """
    Build a mapping from each character index in the plain text (obtained by stripping HTML tags)
    to the corresponding character index in the original HTML string.
    """
    mapping = []
    in_tag = False
    for idx, char in enumerate(html_str):
        if char == "<":
            in_tag = True
        elif char == ">":
            in_tag = False
            continue  # Do not include the '>' itself in plain text
        elif not in_tag:
            # Append the current HTML index for this plain text character.
            mapping.append(idx)
    return mapping


def extract_html_sentences(paragraph):
    """
    Given a BeautifulSoup paragraph tag, extract its inner HTML and plain text,
    use spaCy to split into sentences, and then map back to HTML for each sentence.
    Returns a list of HTML fragments, one per sentence.
    """
    paragraph = convert_square_bracket_refs(paragraph)

    # Get the inner HTML and plain text for the paragraph
    inner_html = paragraph.decode_contents()
    plain_text = paragraph.get_text()

    # Build mapping: plain_text_index -> inner_html_index
    mapping = spacy_sentence_html_mapping(inner_html)

    # Load spaCy English model
    spacy_english = spacy.load("en_core_web_sm")

    # Process plain text with spaCy to get sentence boundaries.
    doc = spacy_english(plain_text)

    sentence_html_fragments = []
    for sent in doc.sents:
        # Get the start and end character positions in plain text.
        start_pt = sent.start_char
        end_pt = sent.end_char

        # Map these positions to the original HTML.
        # Note: The mapping length should equal len(plain_text).
        if start_pt < len(mapping) and end_pt - 1 < len(mapping):
            html_start = mapping[start_pt]
            html_end = mapping[end_pt - 1] + 1  # include last character
            # Extract the corresponding substring from the inner_html.
            sent_html = inner_html[html_start:html_end]
            sentence_html_fragments.append(sent_html)
        else:
            # Fallback: if mapping fails, use plain text.
            sentence_html_fragments.append(sent.text)

    return sentence_html_fragments


def extract_text_and_refs(paragraph):
    """
    Given a paragraph, first splits it into sentences (using Spacy for sentence detection)
    and then for each chunk:
      - Finds all valid reference <a> tags,
      - Extracts the reference numbers,
      - Removes those tags from the chunk's HTML,
      - Cleans up the remaining text,
      - Duplicates the clean text for each reference.

    Returns three lists: context, texts and references.
    """
    # Split the paragraph into chunks based on references and sentence boundaries.
    # chunked_paragraph = split_paragraph_by_reference(paragraph)
    sentences_html = extract_html_sentences(paragraph)
    texts_out = []
    fragments_out = []
    refs_out = []

    # Process each chunk separately.
    for chunk in sentences_html:
        # Parse the chunk HTML.
        chunk_soup = BeautifulSoup(chunk, "html.parser")

        # iterate through the children of chunk_soup determining creating atomic_text and refs. atomic_texts will be a list of text that exists between reference elements and will stop as soon as a reference element appears
        atomic_texts, atomic_refs = extract_atomic_text_and_refs(chunk_soup)

        # Check if the lengths are not the same
        if len(atomic_texts) != len(atomic_refs):
            raise ValueError("Lengths of atomic_texts and atomic_refs are not the same")

        # Find all valid reference <a> tags.
        ref_tags = chunk_soup.find_all(
            "a", class_="usa-link", href=lambda h: h and h.startswith("#R")
        )
        # Filter out only those that have the aria-describedby attribute.
        valid_refs = [tag for tag in ref_tags if tag.has_attr("aria-describedby")]

        # Remove the reference tags from the HTML so they don't appear in the text.
        for tag in valid_refs:
            tag.decompose()

        # Remove empty housekeeping tags (e.g. <sup> left over) that only contain punctuation/whitespace.
        for tag in chunk_soup.find_all():
            # Get the tag's text and strip whitespace.
            text = tag.get_text(strip=True)
            # If the text is non-empty but contains no letters, remove the tag.
            if text and not re.search(r"[A-Za-z]", text):
                tag.decompose()

        # Clean up the remaining text.
        clean_text = clean_up_text(chunk_soup.get_text())

        # For each extracted reference, add the clean text and the reference.
        if len(atomic_texts) == len(atomic_refs):
            for text, ref in zip(atomic_texts, atomic_refs):
                texts_out.append(clean_text)
                fragments_out.append(text)
                refs_out.append(ref)

    # Ensure lengths are identical
    if len(texts_out) != len(refs_out):
        warn(f"Lengths mismatch: {len(texts_out)} texts vs {len(refs_out)} references")
        # Handle the mismatch as needed, e.g., raise an error or log it
        # You can also choose to return empty lists or raise an exception
        return [], []

    return texts_out, fragments_out, refs_out


def extract_atomic_text_and_refs(chunk_soup):
    paragraph_elements = []
    for child in chunk_soup.children:
        if isinstance(child, str):
            paragraph_elements.append(child)
        elif isinstance(child, Tag):
            # Only add <a> tags if they qualify as references.
            if child.name == "a" and "usa-link" in child.get("class", []):
                if contains_reference(child):
                    paragraph_elements.append(child)
            elif child.name == "sup":
                # Process any <a> tags within <sup> that qualify as references.
                for link in child.find_all("a", {"class": "usa-link"}):
                    if contains_reference(link):
                        paragraph_elements.append(link)

    texts, references = [], []
    current_text, just_processed_ref = "", False
    for child in paragraph_elements:
        if contains_reference(child):
            current_ref = extract_reference(child)
            references.append(current_ref)
            if not current_text and just_processed_ref and texts:
                texts.append(texts[-1])
            else:
                current_text = current_text.strip()
                if current_text:
                    texts.append(current_text)
                just_processed_ref = True
                current_text = ""
            continue
        # For non-reference children, clean up the text.
        child = clean_up_text(child)
        if child:
            just_processed_ref = False
            current_text += child
    if len(texts) != len(references):
        warn(
            f"Mismatch in counts: {len(texts)} sentences vs. {len(references)} references in paragraph: {chunk_soup}"
        )
    return texts, references


def contains_bibr_reference(tag):
    """Check if tag contains a valid reference (not figures, supplements, etc.)"""
    href = tag.get("href", "")
    # Valid reference prefixes
    valid_prefixes = ("#R", "#CR")
    # Invalid reference prefixes (figures, supplements, etc.)
    invalid_prefixes = ("#F", "#S", "#Fig", "#MOESM")

    # Check if href starts with any valid prefix
    is_valid = any(href.startswith(prefix) for prefix in valid_prefixes)
    # Check if href starts with any invalid prefix
    is_invalid = any(href.startswith(prefix) for prefix in invalid_prefixes)

    return (
        tag.name == "a"
        and "usa-link" in tag.get("class", [])
        and tag.has_attr("aria-describedby")
        and is_valid
        and not is_invalid
    )


def extract_reference(tag):
    if tag.name != "a":
        tag = tag.find("a", {"class": "usa-link"})
    rid = tag.attrs.get("aria-describedby", "")
    # Match both R and CR followed by numbers
    match = re.search(r"(?:R|CR)(\d+)", rid)
    if match:
        return match.group(1)
    else:
        warn(f"No number found for sentence: {tag}")
        return ""


def merge_style_tags(children):
    merged = []
    buffer = ""
    for child in children:
        if isinstance(child, NavigableString):
            buffer += child
        elif contains_bibr_reference(child):
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(child)
        elif child.name in ["em", "strong", "i"]:
            buffer += child.get_text()
        else:
            buffer += child.get_text()
    if buffer:
        merged.append(buffer.strip())
    return merged


def contains_reference(child):
    if isinstance(child, Tag) and child.name == "a":
        # If it's already an <a> tag, check it directly
        return contains_bibr_reference(child)

    # Otherwise look for nested <a> tags
    link_tags = child.find_all("a", class_="usa-link") if isinstance(child, Tag) else []

    for link in link_tags:
        if contains_bibr_reference(link):
            return True

    return False


def get_references_from_doi(doi):
    """Fetch references for a given DOI from Crossref API"""
    try:
        response = requests.get(f"{CROSSREF_BASE_URL}/{doi}", headers=USER_AGENT_HEADER)
        response.raise_for_status()
        data = response.content
        if data:
            try:
                json_data = json.loads(data)
                return json_data["message"].get("reference", [])
            except json.JSONDecodeError:
                print(f"Error decoding JSON for DOI: {doi}")
    except requests.RequestException as e:
        print(f"Error fetching references for DOI {doi}. Reason: {e}")
    return []


def get_doi_from_reference_number(references, ref_number):
    if ref_number is None:
        print("Reference number is None")
        return None
    try:
        index = int(ref_number) - 1
        return references[index].get("DOI", None)
    except (ValueError, IndexError):
        print(f"Reference number {ref_number} not found")
        return None


def get_unstructured_from_reference_number(references, ref_number):
    if ref_number is not None:
        try:
            index = int(ref_number) - 1
            ref = references[index]
            if "unstructured" in ref:
                return ref.get("unstructured")
            elif len(ref.keys()) > 0:
                return json.dumps(ref)
            else:
                warn(f"Can't find any info for reference number {ref_number}")
                return None
        except (ValueError, IndexError):
            warn(f"Reference number {ref_number} not found")
            return None


def find_section_div(soup):
    # extract the Article part of the HTML
    # this contains all the article content and excludes the headers and footers
    article_section = soup.find(lambda tag: tag.get("aria-label") == "Article content")
    if not article_section:
        warn("Could not identify 'Article content' <section> in HTML")
        return None

    # look for headings that contain Introduction or Background or Main
    article_headings = article_section.find_all("h2", class_="pmc_sec_title")
    intro_sections = []
    for h2 in article_headings:
        if (
            "introduction" in h2.text.lower()
            or "background" in h2.text.lower()
            or h2.text.lower() == "main"
        ):
            if h2.parent and h2.parent.name in ["div", "section"]:
                intro_sections.append(h2.parent)

    # if only one heading detected return it
    if len(intro_sections) == 1:
        return intro_sections[0]

    # show warning if incorrect number of expected Introduction sections
    if len(intro_sections) > 1:
        warn("Detected multiple introduction sections")
        return None

    if len(intro_sections) < 1:
        warn("Detected no introduction sections in article")
        return None

    return None


def find_references_in_paragraph(paragraph):
    """
    Find all references in a paragraph element, handling various formats.
    """
    references = set()

    # Find all anchor tags
    links = paragraph.find_all("a", class_="usa-link")

    for link in links:
        # Get the href attribute and clean it
        href = link.get("href", "")
        if not href:
            continue

        # Remove the '#' prefix
        ref_id = href.lstrip("#")

        # Valid reference prefixes
        valid_prefixes = ("R", "CR")
        # Invalid reference prefixes
        invalid_prefixes = ("F", "S", "Fig", "MOESM")

        # Check if it's a valid reference and not an invalid one
        is_valid = any(ref_id.startswith(prefix) for prefix in valid_prefixes)
        is_invalid = any(ref_id.startswith(prefix) for prefix in invalid_prefixes)

        if is_valid and not is_invalid:
            # Extract the number part (handle both R and CR cases)
            num = re.search(r"(?:R|CR)(\d+)", ref_id)
            if num:
                num = num.group(1)
                try:
                    # Handle ranges (e.g., R6-R10)
                    if "-" in num:
                        start, end = num.split("-")
                        start_num = int(start)
                        end_num = int(end)
                        # Add all numbers in the range
                        for i in range(start_num, end_num + 1):
                            references.add(str(i))
                    else:
                        references.add(num)
                except ValueError:
                    continue

    return sorted(references, key=lambda x: int(x))


# def split_paragraph_by_reference(paragraph):
#     """
#     Splits a paragraph into chunks. A chunk is defined as everything from the
#     beginning until the end of a sentence (an unformatted period) after a valid reference.

#     A valid reference is:
#       - an <a> tag (or within a <sup>) that has "usa-link" in its class,
#       - has an "aria-describedby" attribute, and
#       - its href starts with "#R".

#     Periods that occur inside nested tags are not considered (since we process
#     plain text nodes separately).
#     """
#     # TODO: remove this when we have fixed paragraph splitting
#     # Define the path to save the paragraphs
#     save_path = os.path.join(os.path.expanduser('~/Downloads'), 'paragraphs_to_split.html')

#     # Check if the file exists, if not create it
#     if not os.path.exists(save_path):
#         with open(save_path, 'w') as file:
#             file.write('<html><body></body></html>')

#     # Append the current paragraph to the file
#     with open(save_path, 'a') as file:
#         file.write(str(paragraph) + '\n')
#     # END OF TEMP CODE


#     chunks = []
#     current_chunk = ""
#     after_reference = False  # indicates that we have encountered a valid reference

#     for child in paragraph.contents:
#         if isinstance(child, NavigableString):
#             text = str(child)
#             if after_reference:
#                 # Look for an unformatted period in the plain text.
#                 period_index = text.find(".")
#                 if period_index != -1:
#                     # Include text up to and including the period.
#                     current_chunk += text[:period_index + 1]
#                     # End the current chunk.
#                     chunks.append(current_chunk.strip())
#                     # Continue with any text following the period.
#                     current_chunk = text[period_index + 1:]
#                     after_reference = False
#                 else:
#                     # No period found yet; just accumulate.
#                     current_chunk += text
#             else:
#                 current_chunk += text
#         elif isinstance(child, Tag):
#             if child.name == "sup":
#                 # Check if this <sup> contains any valid reference <a> tag.
#                 valid_ref = False
#                 for a in child.find_all("a", class_="usa-link"):
#                     href = a.get("href", "")
#                     # Valid reference prefixes
#                     valid_prefixes = ("#R", "#CR")
#                     # Invalid reference prefixes
#                     invalid_prefixes = ("#F", "#S", "#Fig", "#MOESM")

#                     is_valid = any(href.startswith(prefix) for prefix in valid_prefixes)
#                     is_invalid = any(href.startswith(prefix) for prefix in invalid_prefixes)

#                     if (a.has_attr("aria-describedby") and is_valid and not is_invalid):
#                         valid_ref = True
#                         break
#                 if valid_ref:
#                     current_chunk += str(child)
#                     after_reference = True
#                 else:
#                     # Not a valid reference block, so add its text.
#                     current_chunk += child.get_text()
#             elif child.name == "a":
#                 # For direct <a> tags.
#                 href = child.get("href", "")
#                 # Valid reference prefixes
#                 valid_prefixes = ("#R", "#CR")
#                 # Invalid reference prefixes
#                 invalid_prefixes = ("#F", "#S", "#Fig", "#MOESM")

#                 is_valid = any(href.startswith(prefix) for prefix in valid_prefixes)
#                 is_invalid = any(href.startswith(prefix) for prefix in invalid_prefixes)

#                 if (child.has_attr("aria-describedby") and is_valid and not is_invalid):
#                     current_chunk += str(child)
#                     after_reference = True
#                 else:
#                     # Not a valid reference (e.g. figure or supplement); just add its text.
#                     current_chunk += child.get_text()
#             else:
#                 # For any other tag, accumulate its text.
#                 current_chunk += child.get_text()

#     # Add any remaining text as a final chunk.
#     if current_chunk.strip():
#         chunks.append(current_chunk.strip())
#     return chunks
