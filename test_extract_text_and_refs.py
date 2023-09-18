import unittest

from bs4 import BeautifulSoup

# Assuming the function is in a file named `extractor.py`
from extract_sentences import extract_text_and_refs


class TestExtractTextAndRefs(unittest.TestCase):
    def get_paragraphs_from_html(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        paragraph = soup.find("p")
        return paragraph

    def test_extract_text_and_refs_named_authors(self):
        html_content = """
<p id="p0055" class="p p-first">We leveraged the PCAWG dataset to characterize ITH across cancer types, including single-nucleotide variants (SNVs), insertions or deletions (indels), SVs, and CNAs as well as subclonal drivers, subclonal selection, and mutation signatures (<a href="#bib4" rid="bib4" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">Alexandrov et&nbsp;al., 2020</a>; <a href="#bib42" rid="bib42" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">Gerstung et&nbsp;al., 2020</a>; <a href="#bib79" rid="bib79" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">Rheinbay et&nbsp;al., 2020</a>; <a href="#bib49" rid="bib49" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">ICGC/TCGA Pan-Cancer Analysis of Whole Genomes Consortium, 2020</a>).</p>
"""
        paragraphs = self.get_paragraphs_from_html(html_content)
        texts, refs = extract_text_and_refs(paragraphs)

        expected_texts = [
            "We leveraged the PCAWG dataset to characterize ITH across cancer types, including single-nucleotide variants (SNVs), insertions or deletions (indels), SVs, and CNAs as well as subclonal drivers, subclonal selection, and mutation signatures",
            "We leveraged the PCAWG dataset to characterize ITH across cancer types, including single-nucleotide variants (SNVs), insertions or deletions (indels), SVs, and CNAs as well as subclonal drivers, subclonal selection, and mutation signatures",
            "We leveraged the PCAWG dataset to characterize ITH across cancer types, including single-nucleotide variants (SNVs), insertions or deletions (indels), SVs, and CNAs as well as subclonal drivers, subclonal selection, and mutation signatures",
            "We leveraged the PCAWG dataset to characterize ITH across cancer types, including single-nucleotide variants (SNVs), insertions or deletions (indels), SVs, and CNAs as well as subclonal drivers, subclonal selection, and mutation signatures",
        ]
        expected_refs = ["4", "42", "79", "49"]

        self.assertEqual(len(refs), 4)
        self.assertEqual(len(texts), 4)
        self.assertEqual(texts, expected_texts)
        self.assertEqual(refs, expected_refs)

    def test_extract_text_and_refs_nature_style(self):
        html_content = """
<p id="Par3" class="p p-first">Previous large-scale sequencing projects have identified many putative cancer genes, but most efforts have concentrated on mutations and copy-number alterations in protein-coding genes, mainly using whole-exome sequencing and single-nucleotide polymorphism arrays<sup><a href="#CR1" rid="CR1" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">1</a>–<a href="#CR4" rid="CR4" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">4</a></sup>. Whole-genome sequencing has made it possible to systematically survey non-coding regions for potential driver events, including single-nucleotide variants (SNVs), small insertions and deletions (indels) and larger structural variants. Whole-genome sequencing enables the precise localization of structural variant breakpoints and connections between distinct genomic loci (juxtapositions). Although previous whole-genome sequencing analyses of modestly sized cohorts have revealed candidate non-coding regulatory driver events<sup><a href="#CR8" rid="CR8" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">8</a>–<a href="#CR15" rid="CR15" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">15</a></sup>, the frequency and functional implications of these events remain understudied<sup><a href="#CR6" rid="CR6" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">6</a>,<a href="#CR7" rid="CR7" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">7</a>,<a href="#CR13" rid="CR13" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">13</a>,<a href="#CR16" rid="CR16" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">16</a>,<a href="#CR17" rid="CR17" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">17</a></sup>.</p>
"""
        paragraphs = self.get_paragraphs_from_html(html_content)
        texts, refs = extract_text_and_refs(paragraphs)
        expected_texts = [
            "Previous large-scale sequencing projects have identified many putative cancer genes, but most efforts have concentrated on mutations and copy-number alterations in protein-coding genes, mainly using whole-exome sequencing and single-nucleotide polymorphism arrays",
            "Previous large-scale sequencing projects have identified many putative cancer genes, but most efforts have concentrated on mutations and copy-number alterations in protein-coding genes, mainly using whole-exome sequencing and single-nucleotide polymorphism arrays",
            "Whole-genome sequencing has made it possible to systematically survey non-coding regions for potential driver events, including single-nucleotide variants (SNVs), small insertions and deletions (indels) and larger structural variants. Whole-genome sequencing enables the precise localization of structural variant breakpoints and connections between distinct genomic loci (juxtapositions). Although previous whole-genome sequencing analyses of modestly sized cohorts have revealed candidate non-coding regulatory driver events",
            "Whole-genome sequencing has made it possible to systematically survey non-coding regions for potential driver events, including single-nucleotide variants (SNVs), small insertions and deletions (indels) and larger structural variants. Whole-genome sequencing enables the precise localization of structural variant breakpoints and connections between distinct genomic loci (juxtapositions). Although previous whole-genome sequencing analyses of modestly sized cohorts have revealed candidate non-coding regulatory driver events",
            "the frequency and functional implications of these events remain understudied",
            "the frequency and functional implications of these events remain understudied",
            "the frequency and functional implications of these events remain understudied",
            "the frequency and functional implications of these events remain understudied",
            "the frequency and functional implications of these events remain understudied",
        ]
        expected_refs = ["1", "4", "8", "15", "6", "7", "13", "16", "17"]

        self.assertEqual(len(refs), 9)
        self.assertEqual(len(texts), 9)
        self.assertEqual(refs, expected_refs)
        self.assertEqual(texts, expected_texts)

    def test_extract_text_and_refs_figure_links_and_refs(self):
        html_content = """
<p id="Par6" class="p p-first">Many protein-coding driver mutations occur in single-site ‘hotspots’. In the PCAWG dataset, only 12&nbsp;single-nucleotide positions were mutated in &gt;1%, and 106&nbsp;in &gt;0.5%, of patients (Extended Data Fig. <a href="/pmc/articles/PMC7054214/figure/Fig5/" target="figure" class="fig-table-link figpopup" rid-figpopup="Fig5" rid-ob="ob-Fig5" co-legend-rid="lgnd_Fig5"><span style="position: relative;text-decoration:none;">​<span class="figpopup-sensitive-area" style="left: -2em;">Fig.1a,</span></span><span>1a</span></a>, <a href="#Sec14" rid="Sec14" class=" sec">Methods</a>). Although protein-coding regions span only about 1% of the genome, 15 out of 50 (30%) of the most frequently mutated sites were well-studied hotspots in cancer genes (<em>KRAS</em>, <em>BRAF</em>, <em>PIK3CA</em>, <em>TP53</em> and <em>IDH1</em>) (Fig. <a href="/pmc/articles/PMC7054214/figure/Fig1/" target="figure" class="fig-table-link figpopup" rid-figpopup="Fig1" rid-ob="ob-Fig1" co-legend-rid="lgnd_Fig1"><span style="position: relative;text-decoration:none;">​<span class="figpopup-sensitive-area" style="left: -2.5em;">(Fig.1a,</span></span><span>1a</span></a>, Extended Data Fig. <a href="/pmc/articles/PMC7054214/figure/Fig5/" target="figure" class="fig-table-link figpopup" rid-figpopup="Fig5" rid-ob="ob-Fig5" co-legend-rid="lgnd_Fig5"><span style="position: relative;text-decoration:none;">​<span class="figpopup-sensitive-area" style="left: -2em;">Fig.1b),</span></span><span>1b</span></a>), along with the two canonical <em>TERT</em> promoter hotspots<sup><a href="#CR6" rid="CR6" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">6</a>,<a href="#CR7" rid="CR7" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">7</a></sup>.</p>
        """
        paragraphs = self.get_paragraphs_from_html(html_content)
        texts, refs = extract_text_and_refs(paragraphs)

        expected_texts = [
            "Many protein-coding driver mutations occur in single-site ‘hotspots’. In the PCAWG dataset, only 12 single-nucleotide positions were mutated in >1%, and 106 in >0.5%, of patients (Extended Data Fig.Although protein-coding regions span only about 1% of the genome, 15 out of 50 (30%) of the most frequently mutated sites were well-studied hotspots in cancer genesandFig.Extended Data Fig.along with the two canonicalpromoter hotspots",
            "Many protein-coding driver mutations occur in single-site ‘hotspots’. In the PCAWG dataset, only 12 single-nucleotide positions were mutated in >1%, and 106 in >0.5%, of patients (Extended Data Fig.Although protein-coding regions span only about 1% of the genome, 15 out of 50 (30%) of the most frequently mutated sites were well-studied hotspots in cancer genesandFig.Extended Data Fig.along with the two canonicalpromoter hotspots",
        ]
        expected_refs = ["6", "7"]
        self.assertEqual(len(refs), 2)
        self.assertEqual(len(texts), 2)
        self.assertEqual(refs, expected_refs)
        self.assertEqual(texts, expected_texts)

    def test_extract_text_and_refs_named_authors_wt_figures(self):
        html_content = """
        <p id="p0060">First, to generate high-confidence calls, we developed ensemble approaches for variant calling, copy number calling, and subclonal reconstruction (<a href="/pmc/articles/PMC8054914/figure/fig1/" target="figure" class="fig-table-link figpopup" rid-figpopup="fig1" rid-ob="ob-fig1" co-legend-rid="lgnd_fig1"><span>Figure&nbsp;1</span></a>A; <a href="#sec4" rid="sec4" class=" sec">STAR Methods</a>). Specifically, to maximize the sensitivity and specificity of calling clonal and subclonal mutations, a consensus approach was adopted, integrating the output of five SNV calling algorithms (<a href="#bib49" rid="bib49" class=" bibr popnode" role="button" aria-expanded="false" aria-haspopup="true">ICGC/TCGA Pan-Cancer Analysis of Whole Genomes Consortium, 2020</a>). Similar approaches were employed for indels and SVs.</p>
        """
        paragraph = self.get_paragraphs_from_html(html_content)
        texts, refs = extract_text_and_refs(paragraph)

        expected_texts = [
            "First, to generate high-confidence calls, we developed ensemble approaches for variant calling, copy number calling, and subclonal reconstructionASpecifically, to maximize the sensitivity and specificity of calling clonal and subclonal mutations, a consensus approach was adopted, integrating the output of five SNV calling algorithms"
        ]
        expected_refs = ["49"]

        self.assertEqual(len(refs), 1)
        self.assertEqual(len(texts), 1)
        self.assertEqual(refs, expected_refs)
        self.assertEqual(texts, expected_texts)

    # def test_extract_text_and_refs_named_authors(self):
    # texts, refs = extract_text_and_refs(paragraph)
    # print(texts)
    # print(refs)


if __name__ == "__main__":
    unittest.main()
