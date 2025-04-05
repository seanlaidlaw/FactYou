import hashlib
import json
import os
import time
from datetime import datetime
from logging import warn

import requests
from bs4 import BeautifulSoup

from factyu.config import (CACHE_DIR, CROSSREF_BASE_URL,
                           EUTILS_SUMMARY_BASE_URL, PMC_BASE_URL,
                           PUBMED_ID_CONV, USER_AGENT_HEADER)


def get_cache_path(pmc_id):
    """Generate a cache file path based on the PMC ID"""
    # Create a hash of the URL to use as the filename
    url = f"{PMC_BASE_URL}/PMC{pmc_id}"
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{url_hash}.html")


def fetch_html_from_PMCID(pmc_id):
    """
    Fetch HTML content for a given PMC ID, using cache if available.

    Args:
        pmc_id (str): The PMC ID of the article to fetch

    Returns:
        bytes: The HTML content of the article, or None if fetch failed
    """
    try:
        url = f"{PMC_BASE_URL}/PMC{pmc_id}"
        # Check if we have a cached version
        cache_path = get_cache_path(pmc_id)
        if os.path.exists(cache_path):
            print(f"Using cached version of PMC{pmc_id}")
            with open(cache_path, "rb") as f:
                return f.read()

        # If not cached, fetch from URL
        response = requests.get(url, headers=USER_AGENT_HEADER)
        response.raise_for_status()  # Raise HTTPError for bad status

        # save the PMC article html for faster retrieval
        with open(cache_path, "wb") as f:
            f.write(response.content)
            # if we had to fetch article from pubmed pause for 1s
            # to avoid overloading server
            time.sleep(1)

        return response.content
    except requests.RequestException as e:
        print(f"Error: Unable to fetch data for PMC{pmc_id}. Reason: {e}")
        return None


def date_str_to_datetime(date_str):
    try:
        return datetime.strptime(date_str, "%Y %b %d").date()
    except ValueError:
        return None


def get_article_date(pmc_id, email):
    params = {
        "db": "pmc",
        "id": pmc_id,
        "tool": "FactYou",
        "email": email,
    }
    try:
        response = requests.get(
            EUTILS_SUMMARY_BASE_URL, headers=USER_AGENT_HEADER, params=params
        )
        response.raise_for_status()
        response_xml = response.content
        if response_xml:
            soup = BeautifulSoup(response_xml, "xml")
            date_str = (
                soup.find("Item", {"Name": "PubDate"}).text
                if soup.find("Item", {"Name": "PubDate"})
                else ""
            )
            return date_str_to_datetime(date_str)
    except requests.RequestException as e:
        print(f"Error fetching article date for PMC{pmc_id}. Reason: {e}")
    return None


def sort_pmc_ids_by_date(pmc_ids, email):
    pmc_date_pairs = [(pmc_id, get_article_date(pmc_id, email)) for pmc_id in pmc_ids]
    filtered_pairs = [pair for pair in pmc_date_pairs if pair[1] is not None]
    sorted_pairs = sorted(filtered_pairs, key=lambda x: x[1])
    sorted_ids = [pair[0] for pair in sorted_pairs]
    return sorted_ids


def get_pmc_from_doi(doi, email):
    params = {"tool": "FactYou", "email": email, "ids": doi}
    try:
        response = requests.get(
            PUBMED_ID_CONV, headers=USER_AGENT_HEADER, params=params
        )
        response.raise_for_status()
        response_xml = response.content
        if response_xml:
            soup = BeautifulSoup(response_xml, "xml")
            error_list = soup.find("ErrorList")
            if not error_list:
                pmc_records = soup.find_all("record")
                pmc_ids = []
                if len(pmc_records) > 0:
                    for record in pmc_records:
                        pmc_ids.append(record.get("pmcid"))
                if len(pmc_ids) == 1:
                    return pmc_ids[0]
                    # pmc_ids = [tag.text for tag in id_list.find_all("Id")]
                    # sorted_pmc_ids = sort_pmc_ids_by_date(pmc_ids, email)
                    # oldest_pmc_id = sorted_pmc_ids[0] if sorted_pmc_ids else None
                    # return oldest_pmc_id
    except requests.RequestException as e:
        print(f"Error fetching PMC ID for DOI {doi}. Reason: {e}")
    return None
