# COMP3009J Information Retrieval

## Large Document Corpus for Programming Assignment: Version 1.2

This file describes the contents of the corpus and associated files, along with their formats.

### Documents
The documents are contained in the [documents](documents) directory. Documents are organised in subdirectories according to their document ID. Each file in a subdirectory contains one document. The document ID is the filename. All files are plain text.

### Standard Queries
The file ``[queries.txt](files/queries.txt)'' contains the standard queries for evaluation.

Each query is on a new line, with the query ID before the query keywords. In total there are 82 queries.

### Relevance Judgments
The file ``[qrels.txt](files/qrels.txt)'' contains the relevance judgments. These are in the [format used by the TREC conference](https://trec.nist.gov/data/qrels_eng/).

Each line has 4 fields, separated by whitespace. The fields are as follows:

1.	The Query ID.
2.	(this field is always 0 and can be ignored)
3.	The Document ID.
4.	The relevance judgment. This is 0 for documents that are judged non-relevant. For judged relevant documents, a higher score means a higher level of relevance.

Any document that does not have a relevance judgment for any query is unjudged.

### Output Format
A sample output file (with randomly chosen documents) is shown in [sample_output.txt](files/sample_output.txt). This is the format your results should be in.

The format for this is also that used by the TREC conference. This file will have 6 fields on each line, which are:

1. The Query ID.
2. The string "Q0" (this is generally ignored)
3. The Document ID.
4. The rank of the document in the results for this query (starting at 1).
5. The similarity score for the document and this query.
6. The name of the run (this should be your UCD student ID number).

### Questions
If you have questions about this corpus, please post in the Brightspace discussion forum, or email me at [david.lillis@ucd.ie](mailto:david.lilli@ucd.ie).