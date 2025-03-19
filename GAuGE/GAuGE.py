from ElectraScorer import ElectraScorer
from MistralQA import MistralQA
from rouge_score import rouge_scorer
import bm25s
import random

class GAuGE:
    def __init__(self, ROUGETYPE = "rouge2", LAMBDA = 10, DOC_DEPTH = 2, DEPTH = 2, no_of_mutations_per_iteration = 12):
        self.ROUGETYPE = ROUGETYPE
        self.RougeScorer = rouge_scorer.RougeScorer([self.ROUGETYPE], use_stemmer = True)
        self.ElectraScorer = ElectraScorer()
        self.MistralQA = MistralQA()
        self.LAMBDA = LAMBDA  # scaling parameter
        self.DOC_DEPTH = DOC_DEPTH  # 𝑡 document sampling depth
        self.DEPTH = DEPTH  # 𝑑 termination depth
        self.no_of_mutations_per_iteration = no_of_mutations_per_iteration  # 𝑚 mutations per iteration

    def generation(self, query):
        # Step1. BM25 檢索
        query_tokens = bm25s.tokenize(query)
        retriever = bm25s.BM25.load("bm25s_msmarcoV2.1_GAuGE", load_corpus = True)
        docs, scores = retriever.retrieve(query_tokens, k = 5)
        top1 = docs[0][0]["text"]
        top2 = docs[0][1]["text"]

        # Step2. Add re-ranking retrieval results to heap
        heap = [(float("-inf"), "")]
        for doc in docs[0]:
            electra = self.ElectraScorer.RunModel(query, doc["text"])
            rouge = self.RougeScorer.score(top1 + " " + top2, doc["text"])
            heap.append((electra + self.LAMBDA * rouge[self.ROUGETYPE][2], doc["text"]))
        heap = sorted(heap)

        # Step3. Genetic Algorithm
        iter = 0
        while True:
            last_heap_depth_score = heap[-1 * self.DEPTH][0]
            # Mutations
            res = []
            for n in range(self.no_of_mutations_per_iteration):
                case = random.randint(1, 3)
                try:
                    if case == 1:
                        # rewrite
                        docid = random.randint(1, 2)
                        temp = self.MistralQA.rewrite(heap[-1 * docid][1])
                        res.append(temp)
                    elif case == 2:
                        # query_write
                        docid = random.randint(1, 2)
                        temp = self.MistralQA.query_rewrite(heap[-1 * docid][1], query)
                        res.append(temp)
                    else:
                        # combine
                        docid1, docid2 = 1, 2
                        temp = self.MistralQA.combind(heap[-1 * docid1][1], heap[-1 * docid2][1], query)
                        res.append(temp)
                except Exception as e:
                    print(e)
                    continue
                iter += 1
    
            # Evaluate new documents
            for DocAfterMutation in res:
                electra = self.ElectraScorer.RunModel(query, DocAfterMutation)
                rouge = self.RougeScorer.score(top1 + " " + top2, DocAfterMutation)
                heap.append((electra + self.LAMBDA * rouge[self.ROUGETYPE][2], DocAfterMutation))
            heap = sorted(heap)
        
            # Termination criteria
            if heap[-1 * self.DEPTH][0] <= last_heap_depth_score:
                break

        print("GAuGE Evolutionary Finished")
        return heap[-1], iter