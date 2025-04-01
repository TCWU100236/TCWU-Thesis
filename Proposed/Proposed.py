from Embedding import Embedder
from Qdrant import Qdrant
from FitnessValue import Fitness
from Generator import MistralQA
import torch.nn.functional as F
import random
import torch

class Proposed:
    def __init__(self, num_pops, top_k, top_p, threshold, termination, collection_name):
        # 實驗超參數
        if top_p > top_k:
            raise ValueError("!!參數設定有錯!!")
        self.num_pops = num_pops    # 種群數量
        self.top_k = top_k    # 單一種群初始數量
        self.top_p = top_p    # 每個種群取 p 個 (p < k)
        self.threshold = threshold    # fitness 門檻值 (暫時用不到)
        self.termination = termination    # GA 停止條件
        self.collection_name = collection_name    # 資料庫名稱

        # 實驗用到的 module
        self.Embedder = Embedder(self.num_pops)
        self.FitnessValue = Fitness()
        self.generator = MistralQA()
        self.Qdrant_DB = Qdrant(collection_name)

    # 初始化種群
    def Init_Population(self, head_index, query_vector, k):
        """
        decription:
            單一種群初始化        
        Args:
            collection_name (str): 向量資料庫名稱
            head_index (int): 注意力頭索引
            query_vector (list): 參考向量
        return:
            candidates (list): 第 head_index 個的 (text, vector)
        """
        candidates = []
        search_result = self.Qdrant_DB.query_by_head(head_index, query_vector, k)

        for point in search_result.points:
            candidates.append((point.payload["text"], point.vector[f"head_{head_index}"]))
        
        return candidates 
    
    # 實作 GA 選擇的輪盤選擇法
    def selection(self, orig_candidates):
        """
        decription:
            輪盤選擇法       
        Args:
            orig_candidates (dict): {e_0: [tensor, fitness]} 種群中個體和個體的適應度
        return:
            (tuple): ("e_i", [tensor, fitness])
        """
        randNum = random.uniform(0, 1)  # 產生隨機數
        totalSum = sum(value[1] for value in orig_candidates.values())  # 全部 fitness 的總和
        percentage = {key: value[1] / totalSum for key, value in orig_candidates.items()}  # 各 e 的 fitness 所佔比例
        # partialSum = dict(zip(percentage.keys(), itertools.accumulate(percentage.values())))  # 累積比例

        # partialSum 比例依序加總直到隨機挑選出來的 randNum 小於等於 partialSum
        # 就把所索引值回傳代表此一輪選擇到這個
        partialSum = 0
        for key in orig_candidates.keys():
            partialSum += percentage[key]
            if randNum <= partialSum:
                return (key, orig_candidates[key])
        return 0
    
    def Find_passages(self, query):
        # Step 1. Initialize populations（初始化種群）
        Populations = {}
        query_multi_head = self.Embedder.generate_embedding(query)    # shape = (batch_size, head_nums, head_dim)

        # 待修改
        for idx, vector in enumerate(query_multi_head[0]):
            Populations[f"population_{idx+1}"] = self.Init_Population(idx+1, vector.tolist(), self.top_k)
        
        # Step 2. GA 演化
        context_vectors = {}
        # 各個 population 各自優化
        for pop, candidates in Populations.items():
            candidates_after_evo = {}
            ref = query_multi_head[0][int(pop[-1])-1]
            for idx, information in enumerate(candidates):
                vector = torch.tensor(information[1], device = "cuda")
                Reference_score = self.FitnessValue.calculate(ref, vector)
                candidates_after_evo[f"e_{idx}"] = [vector, Reference_score]
            # print(candidates_after_evo)

            while len(candidates_after_evo) < self.termination:
                # Selection
                parent_1 = self.selection(candidates_after_evo)
                parent_2 = self.selection(candidates_after_evo)
                # Crossover
                average_tensor = (parent_1[1][0] + parent_2[1][0]) / 2
                child_fit = self.FitnessValue.calculate(ref, average_tensor)
                # evaluate
                # if (child_fit > threshold):
                key = f"e_{len(candidates_after_evo)}"
                candidates_after_evo[key] = [average_tensor, child_fit]
            print(f"{pop} 演化完成")
            # print(candidates_after_evo)
            # print("")

            sorted_candidates_after_evo = sorted(candidates_after_evo.items(), key=lambda x: x[1][1], reverse = True)
            context_vectors[pop] = sorted_candidates_after_evo[:self.top_p]
        # print(context_vectors)

        print("GA 進行完成，且已選出每個 pop 演化後最好之vector")

        # Step 3. 回原始種群找最相似的 top-p 個 candidate
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        res = []
        for pop, best_children in context_vectors.items():
            for child in best_children:
                # print(pop, child)
                max_cos_sim = [0] * 3
                A = child[1][0]
                for init_candidate in Populations[pop]:
                    # print(pop, init_candidate)
                    B = torch.tensor(init_candidate[1]).to(device)
                    cos_sim = F.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0)).item()  # 計算 Cosine Similarity 
                    if (cos_sim > max_cos_sim[2]):
                        max_cos_sim[0], max_cos_sim[1], max_cos_sim[2] = init_candidate[0], init_candidate[1], cos_sim
                res.append(max_cos_sim)
        # print(len(res))

        # Step 4. 將最相似的 vector 的 text 組合成 passages
        passages = ""
        for id, passage in enumerate(res):
            passages += f"{id+1}. {passage[0]}\n"
        return passages

    def generation(self, query):
        passages = self.Find_passages(query)
        response = self.generator.run_model(passages, query)
        return passages, response