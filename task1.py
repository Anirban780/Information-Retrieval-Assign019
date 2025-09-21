import re
import math
from collections import defaultdict
from nltk.stem import PorterStemmer

# --------------------------
# Stopword list (compact)
# --------------------------
STOPWORDS = {
    "a","an","the","is","are","in","on","of","and","to","for","with","by","about","from",
    "as","at","into","much","every","more","less","than","this","that","these","those",
    "between","within","without","over","under","up","down","out","very","be","been","being",
    "was","were","am","will","shall","may","might","must","can","could","should",
    "do","does","did","not","no","nor","but","so","if","because","until","while"
}

stemmer = PorterStemmer()

# --------------------------
# Preprocessing
# --------------------------
def preprocess(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [stemmer.stem(tok) for tok in tokens if tok not in STOPWORDS]
    return tokens

# --------------------------
# Posting Node (Linked List)
# --------------------------
class PostingNode:
    def __init__(self, doc_id, freq=0):
        self.doc_id = doc_id
        self.freq = freq
        self.tf = 0.0
        self.tf_idf = 0.0
        self.next = None
        self.skip = None

    def __repr__(self):
        return f"({self.doc_id}, tfidf={self.tf_idf:.4f})"
    

# --------------------------
# Inverted Index
# --------------------------
class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(lambda:None) # term -> head PostingNode
        self.doc_lengths = {}   # doc_id -> token count
        self.N = 0

    def add_document(self, doc_id, tokens):
        self.N += 1
        self.doc_lengths[doc_id] = len(tokens)
        term_freqs = defaultdict(int)
        for tok in tokens:
            term_freqs[tok] += 1

        for term, freq in term_freqs.items():
            # insert new node in sorted order by doc_id
            head = self.index[term]
            new_node = PostingNode(doc_id, freq)

            if not head or doc_id < head.doc_id:
                new_node.next = head
                self.index[term] = new_node
                continue
            
            prev, curr = None, head
            while curr and curr.doc_id < doc_id:
                prev, curr = curr, curr.next

            prev.next = new_node
            new_node.next = curr

    def compute_tfidf(self):
        for term, head in self.index.items():
            df = self._get_df(head)
            idf = self.N / df
            curr = head

            while curr:
                tf = curr.freq / self.doc_lengths[curr.doc_id]
                curr.tf = tf
                curr.tf_idf = tf * idf
                curr = curr.next

    def _get_df(self, head):
        df, curr = 0, head
        while curr:
            df += 1
            curr = curr.next
        return df
    
    def add_skip_pointers(self):
        for term, head in self.index.items():
            postings = []
            curr = head
            while curr:
                postings.append(curr)
                curr = curr.next

            n = len(postings)
            if n > 2:
                skip_len = int(math.sqrt(n))
                for i in range(0, n, skip_len):
                    if i + skip_len < n:
                        postings[i].skip = postings[i + skip_len]

    def get_postings(self, term):
        return self.index.get(term, None)
    

# --------------------------
# Boolean AND query
# --------------------------
def intersect(p1, p2, use_skips=False):
    result = []
    while p1 and p2:
        if p1.doc_id == p2.doc_id:
            result.append(p1.doc_id)
            p1, p2 = p1.next, p2.next
        
        elif p1.doc_id < p2.doc_id:
            if use_skips and p1.skip and p1.skip.doc_id <= p2.doc_id:
                p1 = p1.skip
            else:
                p1 = p1.next

        else:
            if use_skips and p2.skip and p2.skip.doc_id <= p1.doc_id:
                p2 = p2.skip
            else:
                p2 = p2.next

        return result
    
def get_doc_ids(head):
    ids = []
    curr = head
    while curr:
        ids.append(curr.doc_id)
        curr = curr.next
    return ids

# --------------------------
# Example run
# --------------------------
if __name__ == "__main__":
    # load dataset (40 docs)
    corpus = {}
    with open("corpus.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue   # skip empty or malformed line
            doc_id, sent = parts[0], parts[1]
            corpus[int(doc_id)] = preprocess(sent)

    # build index
    index = InvertedIndex()
    for doc_id, tokens in corpus.items():
        index.add_document(doc_id, tokens)

    index.compute_tfidf()
    index.add_skip_pointers()

    # ------------------------
    # Print FULL inverted index
    # ------------------------
    print("\nFull Inverted Index:")
    for term in sorted(index.index.keys()):
        postings = []
        curr = index.get_postings(term)
        while curr:
            postings.append(f"{curr.doc_id}:{curr.tf_idf:.4f}")
            curr = curr.next
        print(f"{term} -> {postings}")

    # ------------------------
    # Example queries
    # ------------------------
    # Frame 5 queries
    queries = [
        "climate change",
        "renewable energy",
        "air pollution",
        "global warming",
        "green technology"
    ]

    # Run results for first 3 queries
    for q in queries[:3]:
        tokens = preprocess(q)
        if not tokens:
            continue
        print(f"\nQuery: {q} -> {tokens}")

        postings_lists = [index.get_postings(t) for t in tokens if index.get_postings(t)]
        if not postings_lists:
            print("   No results")
            continue

        # start with docIDs from the first term
        result_docs = set(get_doc_ids(postings_lists[0]))
        for pl in postings_lists[1:]:
            result_docs &= set(get_doc_ids(pl))

        print("   Docs:", sorted(result_docs) if result_docs else [])