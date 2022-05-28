import json
import math
import os
import re
import time

from files import porter
from optparse import OptionParser

# Defines the OptionParser to parse the command line arguments
parser = OptionParser()
parser.add_option("-m", "--option")  # -m is the option
options, args = parser.parse_args()
option_dict = vars(options)
mode = option_dict['option']  # mode = 'manual' or 'evaluation'

# Defines the path to the files
UCD_STUDENT_ID = "19206226"
CORPUS_PATH = "./"  # Path to the corpus
DOCUMENTS_PATH = CORPUS_PATH + "documents/"  # Path to the documents
STOPWORDS_PATH = CORPUS_PATH + "files/stopwords.txt"  # Path to the stopwords
RESULT_FILE_NAME = "output.txt"  # Name of the output file
INDEX_FILE_NAME = "index.json"  # Name of the index file
# File paths for test files
SAMPLE_OUTPUT_PATH = CORPUS_PATH + "files/output.txt"  # Path to the sample output file
SAMPLE_STANDARD_QUERY_PATH = CORPUS_PATH + "files/queries.txt"  # Path to the sample standard query file
RELEVANCE_JUDGEMENTS_PATH = CORPUS_PATH + "files/qrels.txt"  # Path to the relevance judgements file
k = 1  # BM25 parameter k
b = 0.75  # BM25 parameter b


def get_files(path):
    """
    Get all the files in the path, for small corpus.
    :param path:
    :return: list(files)
    """
    files = []
    for file in os.listdir(path):
        if file == ".DS_Store":
            continue
        files.append((path + "/" + file, file))  # build file path and file name, and add it to the list
    return files


def clean_terms(terms_data: str) -> list:
    """
    Cleans the terms in the query.
    :param terms_data:
    :return: [list of cleaned terms]
    """
    terms_data = re.compile("[a-z|\']+|[\d.]+", re.I).findall(terms_data)
    # terms_data = [term.strip(string.punctuation) for term in terms_data]
    # terms_data = [clean_term(term) for term in terms_data if clean_term(term) is not None]
    # terms_data = re.sub(r"[^A-Za-z0-9\.\'-]", " ", terms_data)
    # terms_data = re.sub(r"\s{2,}", " ", terms_data)
    # terms_data = terms_data.split()
    # terms_data = [term.strip(string.punctuation) for term in terms_data]
    # compare(data, " ".join(terms_data))
    return terms_data


class CachedStemmer(object):
    """
    A class to cache the stemmed terms.
    Also provide functions to get the stemmed terms.
    """

    def __init__(self):
        self.stemmer = porter.PorterStemmer()  # Initialize the stemmer
        self.cache = {}  # Initialize the cache

    def stem(self, word):
        if word not in self.cache:
            self.cache[word] = self.stemmer.stem(word)  # Stem the word
        return self.cache[word]  # Return the stemmed word


class StopWordFilter(object):
    """
    A class to filter the stop words.
    """

    def __init__(self, path):
        stopwords = {}  # Initialize the stopwords
        with open(path, 'r') as f:
            for line in f:
                stopwords[line.strip()] = 1
        self.stop_words = stopwords  # Set the stopwords

    def filter(self, word_list, stemmer=None):
        if stemmer is not None:
            return [stemmer.stem(word) for word in word_list if self.is_not_stopword(word)]  # Return the filtered words
        return [word for word in word_list if word not in self.stop_words]

    def is_not_stopword(self, term):
        return term not in self.stop_words and len(term) > 0  # Return True if the term is not a stopwords


def compare(str1, str2):
    """
    A Debugging function to compare two strings,
    write the difference to the HTML file.
    :param str1:
    :param str2:
    :return: None
    """
    str2 = str2.split()
    str2 = [f"({term})" for term in str2]
    str2 = " ".join(str2)
    with open("test.html", "a") as f:
        f.write("<p>" + str1 + "</p>")
        f.write("<p>" + str2 + "</p><br>")


cached_stemmer = CachedStemmer()  # Initialize the cached stemmer
stopwords_filter = StopWordFilter(STOPWORDS_PATH)  # Initialize the stopword filter


def dump_index(BM25_Score, enableCompression=False):
    """
    Dumps the index to index.json.
    By default, the index is not compressed, since compression will delay the loading time.
    Compression will reduce the index file size by about 35%.
    :param BM25_Score:
    :param enableCompression:
    :return:
    """
    if not enableCompression:
        # not using my compression algorithm
        data = {
            "BM25_Score": BM25_Score,
        }
    else:
        # using my compression algorithm which minimizes the duplication of terms and docIDs.
        # BUT, it will reduce the speed of the indexing process and loading time.
        print("Compressing..")
        compressID2RealData = {}  # ID to real data mapping
        RealData2CompressedID = {}  # real data to ID mapping
        counter = 0
        for term in BM25_Score:
            for docID in BM25_Score[term]:
                if docID not in RealData2CompressedID:
                    RealData2CompressedID[docID] = counter
                    compressID2RealData[counter] = docID
                    counter += 1
                if term not in RealData2CompressedID:
                    RealData2CompressedID[term] = counter
                    compressID2RealData[counter] = term
                    counter += 1

        compressed_BM25_Score = {}
        for term in BM25_Score:
            # Compress the doc_id
            compressed_doc_id = RealData2CompressedID[term]
            compressID2RealData[compressed_doc_id] = term
            compressed_BM25_Score[compressed_doc_id] = {}
            for doc_id in BM25_Score[term]:
                compressed_term = RealData2CompressedID[doc_id]
                compressID2RealData[compressed_term] = doc_id
                compressed_BM25_Score[compressed_doc_id][compressed_term] = BM25_Score[term][doc_id]
        print("Compression Done.")
        # Now, dump the compressed data
        data = {
            "BM25_Score": compressed_BM25_Score,
            "compressID2RealData": compressID2RealData
        }

    print("\nDumping index...")
    with open(INDEX_FILE_NAME, "w") as f:
        # write the data to the file
        json.dump(data, f)


def load_index(enableCompression=False):
    """
    Loads the index from the file.
    :param enableCompression: False if the index is not compressed.
    :return: index_data
    """
    if os.path.exists("index.json"):
        # we have index.json file, so load it.
        print("Loading index from file...")
        data = json.load(open(INDEX_FILE_NAME, "r", encoding='utf8'))
        if enableCompression:
            # we have compressed index, so decompress it.
            compressed_BM25_Score = data['BM25_Score']
            compressID2RealData = data['compressID2RealData']
            BM25_Score_data = {}
            print("Decompressing...")
            for compressed_term in compressed_BM25_Score:
                term = compressID2RealData[compressed_term]
                BM25_Score_data[term] = {}
                for compressed_doc_id in compressed_BM25_Score[compressed_term]:
                    doc_id = compressID2RealData[compressed_doc_id]
                    BM25_Score_data[term][doc_id] = compressed_BM25_Score[compressed_term][compressed_doc_id]
            print("Decompression Done.")
            data = {
                "BM25_Score": BM25_Score_data,
            }
        # return the data
        return data
    # we don't have index.json file, so return None
    return None


def build_index(path: str) -> dict:
    """
    Builds the index for the given path,
    if the index already exists, it loads it.
    :param path:
    :return:
    """
    # # try to load the index
    enableCompression = False
    saved_data = load_index(enableCompression=enableCompression)
    if saved_data is not None:
        # successfully loaded the index, so return it.
        return saved_data["BM25_Score"]
    # we don't have index, so build it.
    print("Building index...")
    BM25_Score = {}  # BM25_Score[term][doc_id] = score
    document_lengths_data = {}  # document_id -> length, the length of the document
    tf_data = {}  # document_id -> term -> frequency, the frequency of the term in the document
    idf_data = {}  # term -> idf, the inverse document frequency of the term
    avg_doc_length_data = 0  # average document length
    files = get_files(path)
    number_of_documents = len(files)
    current_doc_id = 0
    for file_path, file in files:
        document_lengths_data[file] = 0
        current_doc_id += 1
        # print a progress bar in the terminal
        if current_doc_id % 100 == 0 or current_doc_id == number_of_documents:
            # update the progress every 100 documents, or when we reach the last document
            # if we update the progress every document, the processing speed will be too slow,
            # the "print" function will block the processing.
            ratio = int((current_doc_id / number_of_documents) * 100)
            progress_line = "\rProcessing documents: %d%-5s now: %-20s |%-21s|" % (
                ratio, "%", file, "=" * (ratio // 5) + ">")
            print(progress_line, end="")
        with open(file_path, 'r', encoding='UTF-8') as f:
            doc = f.read()  # read the document
            # terms = [term.lower().strip(string.punctuation) for term in doc.split()]
            doc = doc.lower()  # lower the terms
            terms = clean_terms(doc)  # clean the terms using regex
            freq = {}  # key(term),value(frequency)
            for term in terms:
                if stopwords_filter.is_not_stopword(term) and term != ".":
                    # if the term is not a stopwords and not a punctuation mark,
                    # using my cached stemmer to stem the term.
                    term = cached_stemmer.stem(term)
                    document_lengths_data[file] += 1
                    if term not in freq:
                        # if the term is not in the freq, add it to the freq
                        freq[term] = 1
                        if term in idf_data:
                            # if the term is in the idf, add 1 to the idf
                            idf_data[term] += 1
                        else:
                            # if the term is not in the idf, add it to the idf
                            idf_data[term] = 1
                    else:
                        # if the term is in the freq, add 1 to the freq
                        freq[term] += 1
            tf_data[file] = freq
            avg_doc_length_data += document_lengths_data[file]
    # calculate the average document length
    avg_doc_length_data /= number_of_documents
    # calculate the BM25 score for each document
    # print("\nCalculating BM25 scores...")
    for doc_id in tf_data:
        for term in tf_data[doc_id]:
            part2 = math.log((number_of_documents - idf_data[term] + 0.5) / (idf_data[term] + 0.5), 2)
            sim = (tf_data[doc_id][term] * (k + 1) / (
                    tf_data[doc_id][term] + k * (1 - b) + b * document_lengths_data[
                doc_id] / avg_doc_length_data)) * part2
            if term in BM25_Score:
                BM25_Score[term][doc_id] = sim
            else:
                BM25_Score[term] = {doc_id: sim}
    # save the data to the disk
    dump_index(BM25_Score, enableCompression=enableCompression)
    return BM25_Score


def query(query_data: str, BM25_Score_data: dict, adaptive_result=False) -> list:
    query_tf = {}  # term frequency of the query
    similarity = {}  # similarity of the query
    query_data = query_data.lower()  # lower the terms
    terms = clean_terms(query_data)  # clean the terms using regex
    for term in terms:
        if stopwords_filter.is_not_stopword(term):
            # if the term is not a stopwords, using my cached stemmer to stem the term.
            term = cached_stemmer.stem(term)
            if term not in query_tf:
                # if the term is not in the query_tf, add it to the query_tf
                query_tf[term] = 1
            else:
                # if the term is in the query_tf, add 1 to the query_tf
                query_tf[term] += 1
    for term in query_tf:
        if term in BM25_Score_data:
            # if the term is in the term_index, calculate the similarity of the query
            doc_ids = BM25_Score_data[term]
            for doc_id in doc_ids:
                sim = BM25_Score_data[term][doc_id]
                if sim < 0:
                    sim = 0
                if doc_id in similarity:
                    similarity[doc_id] += sim
                else:
                    similarity[doc_id] = sim
    sort = sorted(similarity.items(), key=lambda d: d[1], reverse=True)
    if adaptive_result:
        # if adaptive_result is True,
        # calculate the highest score and lowest score.
        max_score = sort[0][1]
        lowest_score = sort[-1][1]
        standardize_score = []
        for i in range(len(sort)):
            # standardize the scores
            item_tmp = (sort[i][0], (sort[i][1] - lowest_score) / (max_score - lowest_score))
            standardize_score.append(item_tmp)
        # only return score >= 0.5
        scoreThreshold = 0.6
        result = []
        for i in standardize_score:
            doc_score = i[1]
            if doc_score >= scoreThreshold:
                result.append(i)
        # also limit the result to 50
        return result[:50]
    else:
        # if adaptive_result is False, return the top 15 documents
        sort = sort[:15]
        return sort


def run_queries(BM25_Score: dict):
    result_file = open(RESULT_FILE_NAME, 'w', encoding='UTF-8')
    with open(SAMPLE_STANDARD_QUERY_PATH, 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip("\n")
            line = line.split(" ")
            query_id = line[0]
            line = " ".join(line[1:])
            # run each query and write the result to the result file, using the BM25 model.
            res = query(line, BM25_Score, adaptive_result=True)
            for i in range(len(res)):
                # The Query ID.
                # The string "Q0" (this is generally ignored)
                # The Document ID.
                # The rank of the document in the results for this query (starting at 1).
                # The similarity score for the document and this query.
                # The name of the run (this should be your UCD student ID number).
                doc_id = res[i][0]
                rank = i + 1
                sim = res[i][1]
                result_file.write(
                    " ".join([query_id, "Q0", doc_id, str(rank), str(format(sim, '.4f')), UCD_STUDENT_ID]) + "\n")
    result_file.close()
    print("Written to file:", RESULT_FILE_NAME)


def precision(result: list, rel: dict):
    """
    Calculate the precision of the result.
    """
    correct = 0
    for doc_id in result:
        if doc_id in rel and rel[doc_id]:
            correct += 1
    return correct / len(result)


def recall(result: list, rel: dict):
    """
    Calculate the recall of the result.
    """
    relevant_docs_count = 0
    for doc_id in result:
        if doc_id in rel and rel[doc_id]:
            relevant_docs_count += 1
    rel_count = 0
    for doc_id in rel:
        if rel[doc_id] != 0:
            rel_count += 1
    return relevant_docs_count / rel_count


def Pat10(result: list, rel: dict):
    """
    Calculate the P@10 of the result.
    """
    relevant_docs_count = 0
    for doc_id in result[:10]:
        if doc_id in rel and rel[doc_id]:
            relevant_docs_count += 1
    # For P@N, if there are fewer than N results returned, then it will simply be the same as precision
    number_of_documents = min(len(result), 10)
    return relevant_docs_count / number_of_documents


def R_precision(result: list, rel: dict):
    """
    Calculate the R-precision of the result.
    """
    relevant_docs_count = 0
    rel_count = 0
    for doc_id in rel:
        if rel[doc_id] != 0:
            rel_count += 1
    for doc_id in result[:rel_count]:
        if doc_id in rel and rel[doc_id]:
            relevant_docs_count += 1
    # R-precision always divides by the number of relevant documents available.
    return relevant_docs_count / rel_count


def MAP(result: list, rel: dict):
    """
    Calculate the MAP of the result.
    """
    relevant_docs_count = 0
    rel_count = 0
    for doc_id in rel:
        # 0 means non-relevant
        if rel[doc_id] != 0:
            rel_count += 1
    score = 0
    for index, doc_id in enumerate(result):
        rank = index + 1
        if doc_id in rel and rel[doc_id]:
            relevant_docs_count += 1
            # Score for each item
            p = relevant_docs_count / rank
            score += p
    # Average
    score /= rel_count
    return score


def b_pref(result: list, rel: dict):
    """
    Calculate the b-pref of the result.
    This function is different in small/large corpus.
    Because the small corpus does not have unjudged documents.
    """
    score = 0
    # count of relevant documents
    r_count = len(rel)
    # count of non-relevant documents
    n_count = 0
    # for doc_id in rel:
    #     if rel[doc_id] != 0:
    #         r_count += 1
    for doc_id in result:
        # in small corpus: a doc is not in the rel means it is not relevant.
        # in large corpus: a doc will have 0 score if it is not relevant.
        # if doc_id in rel and not rel[doc_id]:
        #     n_count += 1
        if doc_id not in rel:
            n_count += 1
        if doc_id in rel and rel[doc_id]:
            # calculate score
            tmp_score = (1 - (n_count / r_count))
            # the min score of b-pref is 0
            if tmp_score < 0:
                score += 0
            else:
                score += tmp_score
    score /= r_count
    return score


def NDCG(result: list, rel: dict):
    """
    Calculate the NDCG of the result.
    """
    DCG = []
    IDCG = []
    # calculate DCG and IDCG
    for index, doc_id in enumerate(result):
        rank = index + 1
        IG = rel.get(doc_id, 0)
        if index == 0:
            DCG.append(IG / 1)
        else:
            DCG.append(IG / math.log(rank, 2) + DCG[index - 1])
    # Sort rel by value
    rel = sorted(rel.values(), key=lambda x: x, reverse=True)
    for index, value in enumerate(rel):
        rank = index + 1
        IG = value
        if index == 0:
            IDCG.append(IG / 1)
        else:
            IDCG.append(IG / math.log(rank, 2) + IDCG[index - 1])
    # Get the highest index we have in this two lists.
    # if we have more than 10 element in both of these lists, then
    # only use the 10-th element to calculate the NDCG.
    # If the number of returned documents(e.g. 8) is < than relevance(e.g. 10).
    # then we calculate the NDCG according to the min index of returned documents,
    # e.g. DCG8/IDCG8
    if len(DCG) < len(IDCG):
        max_index = len(DCG) - 1
        if max_index > 9:
            max_index = 9
        return DCG[max_index] / IDCG[max_index]
    else:
        # len(DCG) >= len(IDCG)
        # if the number of returned documents(e.g. 8) is > than relevance(e.g. 6).
        # then we use the last element in relevance to calculate the NDCG.
        # since IDCG@6 = IDCG@8 do it doesn't really matter which to use.
        IDCG_score = IDCG[:10][-1]
        DCG_score = DCG[:10][-1]
        return DCG_score / IDCG_score


def evaluate(result_file_name: str):
    result = {}  # To be load with query result
    rel = {}  # To be load with relevance judgements
    with open(result_file_name, 'r', encoding='UTF-8') as f:
        # Read result file
        for line in f:
            line = line.strip("\n")
            line = line.split(" ")
            query_id = line[0]
            doc_id = line[2]
            # add to result dict
            if query_id not in result:
                result[query_id] = [doc_id]
            else:
                result[query_id].append(doc_id)
    with open(RELEVANCE_JUDGEMENTS_PATH, 'r', encoding='UTF-8') as f:
        # Read relevance judgements file
        for line in f:
            # line = line.strip("\n")
            line = line.split()
            query_id = line[0]
            doc_id = line[2]
            score = int(line[3])
            # add to rel dict
            if query_id not in rel:
                rel[query_id] = {doc_id: score}
            else:
                rel[query_id][doc_id] = score

    # Calculate metrics
    precision_score = 0
    recall_score = 0
    Pat10_score = 0
    R_precision_score = 0
    MAP_score = 0
    B_pref_score = 0
    NDCG_score = 0

    for query_id in result.keys():
        # get result and rel line
        result_line = result[query_id]
        rel_line = rel[query_id]
        # calculate metrics
        precision_score += precision(result_line, rel_line)
        recall_score += recall(result_line, rel_line)
        Pat10_score += Pat10(result_line, rel_line)
        R_precision_score += R_precision(result_line, rel_line)
        MAP_score += MAP(result_line, rel_line)
        B_pref_score += b_pref(result_line, rel_line)
        NDCG_score += NDCG(result_line, rel_line)

    # Calculate average
    precision_score /= len(result)
    recall_score /= len(result)
    Pat10_score /= len(result)
    R_precision_score /= len(result)
    MAP_score /= len(result)
    B_pref_score /= len(result)
    NDCG_score /= len(result)
    # Print metrics
    print(f'Precision    ({precision_score})')
    print(f'Recall       ({recall_score})')
    print(f'Precision@10 ({Pat10_score})')
    print(f'R-Precision  ({R_precision_score})')
    print(f'MAP          ({MAP_score})')
    print(f'b_pref       ({B_pref_score})')
    print(f'NDCG_score   ({NDCG_score})')


def auto_evaluate(BM25_Score: dict):
    """
    Evaluate the metrics of the my program.
    :param BM25_Score:
    :return:
    """
    start_time = time.process_time()
    # run all the queries
    run_queries(BM25_Score)
    end_time = time.process_time()
    print(f"Query cost: {end_time - start_time} s")

    start_time = time.process_time()
    # evaluate the result
    evaluate(RESULT_FILE_NAME)
    end_time = time.process_time()
    print(f"Evaluate cost: {end_time - start_time} s")


start_time = time.process_time()
# build index
BM25_Score = build_index(DOCUMENTS_PATH)
end_time = time.process_time()
print(f"Build or loading index cost: {end_time - start_time} s")

if mode == "manual":
    # manual mode, user can input queries
    query_id = 0
    while True:
        input_data = input("\nPlease input query.\n>>> ")
        if input_data.lower() == "exit" or input_data.lower() == "quit":
            break
        else:
            start_time = time.process_time()
            sorted_result = query(input_data, BM25_Score, adaptive_result=False)
            end_time = time.process_time()
            print(f"Time <Query>: {end_time - start_time}")
            query_id += 1
            print(f"Query {query_id}: {input_data}")
            line = "%10s %10s %30s %10s %20s %20s" % ("Query ID", "Q0", "Doc ID", "Ranking", "Score", "Student ID")
            print(line)
            for ranking, i in enumerate(sorted_result):
                document_id = i[0]
                score = i[1]
                line = "%10d %10s %30s %10d %20f %20s" % (query_id, "Q0", document_id, ranking + 1, score, "19206226")
                print(line)

elif mode == "evaluation":
    # evaluation mode, auto evaluate the metrics
    auto_evaluate(BM25_Score)
else:
    # Default mode, run queries
    auto_evaluate(BM25_Score)
