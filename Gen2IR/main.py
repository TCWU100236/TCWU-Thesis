from Gen2IR import Gen2IR
from datasets import load_dataset
from rouge_score import rouge_scorer
import pandas as pd
import time
import os

filepath = r"../Experiments/Gen2IR_test.csv"
# 檢查檔案是否存在
if os.path.isfile(filepath):
    savedf = pd.read_csv(filepath)
else:
    savedf = pd.DataFrame(columns = ["QID", "Query", "Gen2IR_Gen", "Answer", "Precision", "Recall", "Iteration", "Time"])

# 實驗 module
ds = load_dataset("microsoft/ms_marco", "v2.1")
ROUGETYPE = "rouge2"
RougeScorer = rouge_scorer.RougeScorer([ROUGETYPE], use_stemmer = True)

# 實驗超參數
DOC_DEPTH = 2  # 𝑡 document sampling depth
DEPTH = 2  # 𝑑 termination depth
no_of_mutations_per_iteration = 12  # 𝑚 mutations per iteration

for i in range(30, 40):
    query = ds["train"]["query"][i]
    qid = ds["train"]["query_id"][i]
    answer = ds["train"]["answers"][i]
    print(query)

    Gen2IR_method = Gen2IR(DOC_DEPTH, DEPTH, no_of_mutations_per_iteration)
    Start = time.time()
    response, iteration = Gen2IR_method.generation(query)
    End = time.time()
    print(response[1])

    # Save result to csv file
    score = RougeScorer.score(response[1], answer[0])
    newdf = pd.DataFrame([{"QID": qid,
                            "Query": query,
                            "Gen2IR_Gen": response[1],
                            "Answer": answer[0],
                            "Precision": score[ROUGETYPE].precision,
                            "Recall": score[ROUGETYPE].recall,
                            "Iteration": iteration,
                            "Time": End - Start}])
    savedf = pd.concat([savedf, newdf], ignore_index = True)
    
savedf.to_csv(filepath, index = False)
print("該 batch 實驗結果儲存完畢!!")
