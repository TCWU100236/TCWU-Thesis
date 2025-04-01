from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn.functional as F
import torch
import random

class Embedder:
    def __init__(self, n, model_name = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.self_attn_heads_output = []  # 儲存每一個 heads 的輸出

        if n > self.config.num_attention_heads:
            raise ValueError("自注意力頭數量超過嵌入模型本身參數")
        self.n = n
        self.selected_head = random.sample(range(self.config.num_attention_heads), n)  # 超參數: 隨機挑選 n 個 head 的輸出

    def mean_pooling(self, attn_outputs, attention_mask):
        # 擴展 attention_mask 使其與 attn_output 尺寸匹配
        attention_mask_expanded = attention_mask.unsqueeze(-1).unsqueeze(-1).expand(attn_outputs.size())
    
        # 計算 mean pooling，忽略填充 (padding) 部分
        mean_pooled = torch.sum(attn_outputs * attention_mask_expanded, dim=1) / torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        return mean_pooled

    def generate_embedding(self, text):
        
        def hook(module, input, output):
            # output 是形狀為 (batch_size, seq_length, hidden_size) 的 tensor
            # 將其重新 shape 為 (batch_size, seq_length, num_heads, head_dim)
            num_heads = self.config.num_attention_heads  # 確認 head 數量
            head_dim = self.config.hidden_size // self.config.num_attention_heads  # 單個 head 的維度
            reshaped_output = output.view(output.size(0), output.size(1), num_heads, head_dim)
            self.self_attn_heads_output.append(reshaped_output)

        self.self_attn_heads_output.clear()  # 清空之前的儲存結果
        last_ecoder_layer = self.model.encoder.layer[-1] # 取得最後一層的 Ecoder Layer
        hook_handle = last_ecoder_layer.attention.output.register_forward_hook(hook)

        # 分詞
        encoded_input = self.tokenizer(text, padding = True, truncation = True, return_tensors = "pt")
        encoded_input = {k : v.to(self.device) for k, v in encoded_input.items()}

        # 執行前向傳播
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # 使用 mean_pooling 函數計算每個 head 的嵌入
        head_mean_embeddings = self.mean_pooling(self.self_attn_heads_output[0], encoded_input["attention_mask"])
        # 完成後移除 hook
        hook_handle.remove()

        if self.n != self.config.num_attention_heads:
            res = torch.stack([head_mean_embeddings[0][i].cpu() for i in self.selected_head])
            res = res.unsqueeze(0)  # 在第 0 維增加一個維度
            return res.cpu()
        
        return head_mean_embeddings.cpu()
    
# num_heads = 12  # 超參數: 自注意頭數量 (初始種群數量)
# Embed = Embedder(num_heads)
# text = "我會披星戴月的想你，我會奮不顧身的前進，遠方煙火越來越唏噓，凝視前方身後的距離"
# embedding = Embed.generate_embedding(text)
# print(embedding.shape)
# print(embedding)