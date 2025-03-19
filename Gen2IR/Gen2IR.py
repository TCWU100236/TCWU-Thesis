from ElectraScorer import ElectraScorer
from MistralQA import MistralQA
import random
import bm25s

class Gen2IR:
    def __init__(self, DOC_DEPTH = 2, DEPTH = 2, no_of_mutations_per_iteration = 2):
        self.ElectraScorer = ElectraScorer()
        self.MistralQA = MistralQA()
        self.DOC_DEPTH = DOC_DEPTH  # 𝑡 document sampling depth
        self.DEPTH = DEPTH  # 𝑑 termination depth
        self.no_of_mutations_per_iteration = no_of_mutations_per_iteration  # 𝑚 mutations per iteration

    def generation(self, query):
        # Step1. BM25 檢索
        query_tokens = bm25s.tokenize(query)
        retriever = bm25s.BM25.load("bm25s_msmarcoV2.1_Gen2IR", load_corpus = True)
        docs, scores = retriever.retrieve(query_tokens, k = 5)

        # Step2. Add re-ranking retrieval results to heap
        heap = [(float("-inf"), "")]
        for doc in docs[0]:
            score = self.ElectraScorer.RunModel(query, doc["text"])
            heap.append((score, doc["text"]))
        heap = sorted(heap)

        # Step3. Genetic Algorithm
        iter = 0
        while True:
            last_heap_depth_score = heap[-1 * self.DEPTH][0]
            # Step3.1 Mutations
            res = []
            for n in range(self.no_of_mutations_per_iteration):
                case = random.random()
                try:
                    if case <= 0.33:
                        # rewrite
                        docid = int(random.random() * 100) % self.DOC_DEPTH + 1
                        res.append(self.MistralQA.rewrite(heap[-1 * docid][1]))
                    elif case <= 0.66:
                        # query_write
                        docid = int(random.random() * 100) % self.DOC_DEPTH + 1
                        res.append(self.MistralQA.query_rewrite(heap[-1 * docid][1], query))
                    else:
                        # combine
                        docid1 = int(random.random() * 100) % self.DOC_DEPTH + 1
                        docid2 = int(random.random() * 100) % self.DOC_DEPTH + 1
                        if docid1 == docid2:
                            if docid1 == self.DOC_DEPTH:
                                docid2 -= 1
                            else:
                                docid2 += 1
                        res.append(self.MistralQA.combind(heap[-1 * docid1][1], heap[-1 * docid2][1], query))
                    iter += 1
                except Exception as e:
                    print(e)
                    continue
            
            # Step3.2 Evaluate new documents
            for DocAfterMutation in res:
                fitness = self.ElectraScorer.RunModel(query, DocAfterMutation)
                heap.append((fitness, DocAfterMutation))
            heap = sorted(heap)

            # Step3.3 Termination criteria
            if heap[-1 * self.DEPTH][0] <= last_heap_depth_score:
                break

        print("Gen2IR Evolutionary Finished")
        return heap[-1], iter