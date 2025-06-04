#!/usr/bin/env python3

from factyu import export_sentence_analysis_to_tsv

def main():
    # Export the analysis to a TSV file in the current directory
    output_file = "sentence_analysis.tsv"
    print(f"Exporting sentence analysis to {output_file}...")
    export_sentence_analysis_to_tsv(output_file)
    print("Export complete!")

if __name__ == "__main__":
    main() 