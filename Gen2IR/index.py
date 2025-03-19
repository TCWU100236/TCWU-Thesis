"""
    建立 Gen2IR 方法的 BM25 索引
"""

from datasets import load_dataset
import bm25s

# Create your corpus here
ds = load_dataset("microsoft/ms_marco", "v2.1")
passages = ds["train"]["passages"]
corpus = []
for passage in passages:
    for passage_text in passage["passage_text"]:
        corpus.append(passage_text)

# Tokenize the corpus and index it
corpus_tokens = bm25s.tokenize(corpus)
retriever = bm25s.BM25(corpus = corpus, method = "robertson")
retriever.index(corpus_tokens)

# Save index for later retrieval...
retriever.save("bm25s_msmarcoV2.1_Gen2IR")
print("索引建立完成!!")