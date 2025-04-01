import json
from datasets import load_dataset

dataset = dict()
dataID, i = 0, 0
ds = load_dataset("microsoft/ms_marco", "v2.1")
temp = "No Answer Present."
while len(dataset) < 30:
    data_format = dict()
    if (ds["train"]["query_type"][i] == "DESCRIPTION") and (temp not in ds["train"]["answers"][i]) and (ds["train"]["wellFormedAnswers"][i] != []):
        dataID += 1
        data_format["Qid"] = ds["train"]["query_id"][i]
        data_format["Query"] = ds["train"]["query"][i]
        data_format["Query_type"] = ds["train"]["query_type"][i]
        data_format["Answers"] = ds["train"]["answers"][i]
        data_format["wellFormedAnswers"] = ds["train"]["wellFormedAnswers"][i]
        dataset[dataID] = data_format
    i += 1
print(i)

try:
    with open("./dataset_update_test.json", mode = "w") as f:
        f.write(json.dumps(dataset))
    print("~~資料集創建成功~~")
except Exception as e:
    print(e)