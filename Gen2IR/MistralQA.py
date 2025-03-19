from transformers import AutoTokenizer, AutoModelForCausalLM
import re

class MistralQA:
    def __init__(self, model_name = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def query_rewrite(self, passage, query):
        sys_message = """You are an expert editor whose job is to re-write the passage to better answer the question.\nYour response must follow this format:```【Response】:...```\nFor example, given the passage: "【Passage】: Banana is the best fruit." and the query:【Query】:"What's your favorite fruit?"\nYour response would look like so: "【Response】: Bananas are the most nutritious food."\nLet's get started. The passage and user's query are as follows."""
        inputs = f"<s> [INST] {sys_message} [/INST]\n【Passage】: {passage}\n【Query】: {query}\n【Response】: "
        result = self.run_model(inputs)
        return self.output_format(result)

    def rewrite(self, passage):
        sys_message = """You are an expert editor whose job is to re-write the passage.\nYour response must follow this format:```【Response】:...```\nFor example, given the passage: "【Passage】: Banana is the best fruit."\nYour response would look like so: "【Response】: Bananas are the most nutritious food."\nLet's get started. The passage is as follows."""
        inputs = f"<s> [INST] {sys_message} [/INST]\n【Passage】: {passage}\n【Response】: "
        result = self.run_model(inputs)
        return self.output_format(result)

    def combind(self, passage1, passage2, query):
        sys_message = """You are an expert editor whose job is to combine ideas from both Passage1 and Passage2 to answer the question.\nYour response must follow this format:```【Response】:...```\nFor example, given the passage1: "【Passage1】: Banana is the best fruit.", passage2: "【Passage2】: Bananas are the most nutritious food." and the query:【Query】:"What's your favorite fruit?"\nYour response would look like so: "【Response】: Banana is my favorite fruit and it brings me many benefits."\nLet's get started. The passages and user's query are as follows."""
        inputs = f"<s> [INST] {sys_message} [/INST]\n【Passage1】: {passage1}\n【Passage2】: {passage2}\n【Query】: {query}\n【Response】: "
        result = self.run_model(inputs)
        return self.output_format(result)
        
    def run_model(self, inputs):
        input_ids = self.tokenizer(inputs, return_tensors="pt")
        output_id = self.model.generate(**input_ids, max_new_tokens = 256, do_sample = True)
        result = self.tokenizer.decode(output_id[0].tolist(), skip_special_tokens=True)
        return result
    
    def output_format(self, result):
        # 找到第一個【Response】: 之後的所有內容
        first_match = re.findall(r"【Response】:\s*(.*)", result, re.DOTALL)
        if first_match:
            first_match_text = first_match[-1].strip()  # 取得第一個【Response】: 之後的部分
        
            # 在第一個【Response】: 之後的部分搜尋所有【Response】:
            second_matches = re.findall(r"【Response】:\s*(.*)", first_match_text, re.DOTALL)
            if second_matches:
                second_response_text = second_matches[-1].strip()  # 取第二個【Response】: 之後的內容

                response_match = re.findall(r"【Response】:\s*(.*)", second_response_text, re.DOTALL)
                if response_match:
                    final_response_text = response_match[-1].strip()  # 取第二個【Response】: 之後的內容
                    return final_response_text