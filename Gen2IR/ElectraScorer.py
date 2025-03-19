from transformers import AutoTokenizer, ElectraForSequenceClassification
import torch

class ElectraScorer:
    def __init__(self, model_name = "crystina-z/monoELECTRA_LCE_nneg31"):
        """
        Args:
            - model_name (str): HuggingFace 模型名稱
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
        self.model = ElectraForSequenceClassification.from_pretrained(model_name).eval().to(self.device)

    def RunModel(self, query, text):
        """
        Args:
            - query (str): 使用者問題
            - text (str): 相關上下文
        return:
            - score (numpy): 相關性分數
        """
        inps = self.tokenizer(query, text, padding = True, truncation = True, return_tensors = 'pt')
        inps = {k : v.to(self.device) for k, v in inps.items()}

        with torch.no_grad():
            score = self.model(**inps).logits[:, 1].cpu().detach().numpy()
        return score