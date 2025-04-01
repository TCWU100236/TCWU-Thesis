from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

class MistralQA:
    def __init__(self, model_name = "mistralai/Mistral-7B-Instruct-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def input_format(self, passages, query):
        # sys_message = """You are an expert editor whose job is to combine ideas from passages to answer the query.\nFor example, given the passages: ```【Passages】: 1.Banana is the best fruit.``` and the query:```【Query】: What's your favorite fruit?```\nAnd your response would look like so: ```【Response】: Banana is my favorite fruit and it brings me many benefits.```\nNote that your response must follow this format:```【Response】: ```\nLet's get started. The passages and query are as follows."""
        sys_message = "You are an expert editor whose job is to synthesize information from given passages to accurately answer the query.\n\n## Instructions:\n- Use the provided passages as the primary source of information.\n- You may enhance the response by adding relevant background knowledge, logical reasoning, or clarification to improve accuracy, but do not contradict or misinterpret the passages.\n- Ensure the response is natural, coherent, and concise.\n- The response must follow this exact format: ```【Response】: <your response here>```\n\n## Example:\n### Passages:\n1. Transformer-based models have achieved state-of-the-art performance in various NLP tasks due to their ability to capture long-range dependencies.\n2. However, training such models requires significant computational resources, which can limit their accessibility.\n\n### Query:\nWhy are transformer-based models widely used in NLP?\n\n### Correct Response:\n```【Response】: Transformer-based models are widely used in NLP because they achieve state-of-the-art performance by capturing long-range dependencies. Additionally, their self-attention mechanism allows for efficient contextual understanding, making them particularly effective for tasks like machine translation and question answering. However, their high computational requirements may limit accessibility, necessitating optimization techniques such as model pruning or knowledge distillation.```\n\nNow, let's get started. Generate your responses based on the passages and query below."
        inputs = f"<s> [INST] {sys_message} [/INST]\n\n【Passages】:\n{passages}\n【Query】: {query}\n\n【Response】: "
        return inputs

    def output_format(self, result):
        # 找到第一個【Response】: 之後的所有內容
        first_match = re.findall(r"【Response】:\s*(.*)", result, re.DOTALL)
        if first_match:
            first_match_text = first_match[-1].strip()  # 取得第一個【Response】: 之後的部分
        
            # 在第一個【Response】: 之後的部分搜尋所有【Response】:
            second_matches = re.findall(r"【Response】:\s*(.*)", first_match_text, re.DOTALL)
            if second_matches:
                second_response_text = second_matches[-1].strip()  # 取第二個【Response】: 之後的內容

                # 在第二個【Response】: 之後的部分搜尋所有【Response】:
                response_matches = re.findall(r"【Response】:\s*(.*)", second_response_text, re.DOTALL)
                if response_matches:
                    last_response_text = response_matches[-1].strip()  # 取最後一個【Response】: 之後的內容
                    return last_response_text
    
    def run_model(self, passages, query):
        inputs = self.input_format(passages, query)
        print(inputs + "\n")
        input_ids = self.tokenizer(inputs, return_tensors="pt")
        output_id = self.model.generate(**input_ids, max_new_tokens = 256, do_sample = True)
        result = self.tokenizer.decode(output_id[0].tolist(), skip_special_tokens = True)
        # print(result)
        return self.output_format(result)

# query = "你今天好美喔~"
# passages = "我騙你的"
# model = MistralQA()
# response = model.run_model(passages, query)
# print(response)