from Proposed import Proposed
from rouge_score import rouge_scorer
from datasets import load_dataset
import json
import time

# 載入資料集
# ds = load_dataset("microsoft/ms_marco", "v2.1")
with open("../Experiments/Dataset/dataset_update.json", mode = "r") as f:
        datas = json.load(f)
qid = datas["1"]["Qid"]
query = datas["1"]["Query"]
answer = datas["1"]["Answers"][0]

# 評估指標
ROUGETYPE = "rouge2"
RougeScorer = rouge_scorer.RougeScorer([ROUGETYPE], use_stemmer = True)

# 超參數設定
num_pops = 12       # 初始種群數量 (1~12)
top_k = 5          # 初始檢索數量 (不限定 正整數)
top_p = 1           # 每個種群取前 p 個最相關的 context
threshold = 0.5     # fitness value 門檻值 (暫時用不到)
termination = 50    # GA 終止條件
collection_name = "qdrant_msmarcoV2.1_Proposed_1000"  # 資料庫名稱

result = dict()
Proposed_method = Proposed(num_pops, top_k, top_p, threshold, termination, collection_name)
Start = time.time()
passages, response = Proposed_method.generation(query)
End = time.time()
print(response)

score = RougeScorer.score(response, answer[0])
result["Qid"] = qid
result["Query"] = query
result["Proposed_Gen"] = response
result["Passages"] = passages
result["Answer"] = answer 
result["Precision"] = score[ROUGETYPE].precision
result["Recall"] = score[ROUGETYPE].recall
result["Num_pops"] = num_pops
result["Top_k"] = top_k
result["Top_p"] = top_p
result["Termenation"] = termination
result["Time"] = time
for k, v in result.items():
    print()
    print(k, v)