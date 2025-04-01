import torch
import numpy as np
from transformers import AutoTokenizer, ElectraForSequenceClassification

class Fitness:
    def __init__(self):
        # 初始化 tokenizer 和 model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fit_linear_layer = torch.nn.Linear(32, 768).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
        self.model = ElectraForSequenceClassification.from_pretrained("crystina-z/monoELECTRA_LCE_nneg31").eval().to(self.device)

    def calculate(self, query_vector, data_vector):
        # query_vector = torch.tensor(query_vector).view(1, 32).to(device)
        query_vector = query_vector.to(self.device)
        # data_vector = torch.tensor(data_vector).view(1, 32).to(device)
        data_vector = data_vector.to(self.device)
        
        # Hook 函數，用於攔截和修改輸入
        def encoder_input_hook(module, inputs):
            inputs[0][0][0] = self.fit_linear_layer(query_vector)
            inputs[0][0][1] = self.fit_linear_layer(data_vector)
            return inputs

        # 註冊 Hook 到第一個 Encoder 層
        encoder_layer = self.model.electra.encoder.layer[0]  # ElectraEncoder 的第一層
        hook_handle = encoder_layer.register_forward_pre_hook(encoder_input_hook)

        # 模擬一個空的 token 輸入作為佔位符（會被 Hook 攔截）
        query = ""
        text = ""
        encoded_input = self.tokenizer(query, text, padding = True, truncation = True, return_tensors = "pt")
        encoded_input = {k : v.to(self.device) for k, v in encoded_input.items()}
        
        with torch.no_grad():
            # score = model(**encoded_input).logits[:, 1].cpu().detach().numpy()
            logits = self.model(**encoded_input).logits[:, 1]  # 提取 logits
            score = torch.sigmoid(logits).cpu().detach().numpy()  # 正規化
    
        # 移除 Hook
        hook_handle.remove()
        return score
    
# import random
# query_vector = torch.tensor([random.uniform(0, 1) for _ in range(32)])
# data_vector = torch.tensor([random.uniform(0, 1) for _ in range(32)])
# FitnessValue = Fitness()
# relevance = FitnessValue.calculate(query_vector, data_vector)
# print(relevance)