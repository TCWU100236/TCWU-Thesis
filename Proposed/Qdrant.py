from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, VectorParams, Distance

class Qdrant:
    def __init__(self, collection_name):
        self.client = QdrantClient(url = "http://localhost:6333")
        self.collection_name = collection_name
        self.create_collection()

    # 建立資料庫
    def create_collection(self):
        """
        Args:
            collection_name (str): 向量資料庫名稱
        """
        # 如果 collection 不存在，就建立一個新 collection
        if self.client.collection_exists(collection_name = self.collection_name):
            print(f"collection: {self.collection_name} 已存在!!")
        else:
            try:
                self.client.create_collection(
                    collection_name = self.collection_name,
                    vectors_config = {
                        "head_" + str(i+1): VectorParams(
                            size = 32, # vector dimension
                            distance = Distance.COSINE, # similarity calculation
                        ) for i in range(12)
                    },
                )
                return f"collection: {self.collection_name} 建立完成!!"
            except Exception as e:
                print(e)

    # 新增資料
    def add_data(self, id, doc, embeddings):
        """
        Args:
            collection_name (str): 資料庫名稱
            id (int): 文檔的 primary key
            doc (str): 文檔的 text
            embeddings (list): 每一個 head 的 vector
        """
        try:
            self.client.upsert(
                collection_name = self.collection_name,
                wait = True,
                points = [PointStruct(id = id,
                                        vector = {"head_" + str(i+1) : embeddings[0][i] for i in range(12)},
                                        payload = {"text" : doc})],
                )
        except Exception as e:
            print(e)
    
    # 查詢
    def query_by_head(self, head_index, query_vector, top_k = 5):
        """
        Args:
            collection_name (str): 資料庫名稱
            head_index (int): 注意力頭的 id 0~11
            query_vector (list): 問題向量
            top_k (int): 返回前 top_k 筆資料
        return:
            search_result (object): 搜尋結果
        """
        # 建構 head 名稱，例如 'head_1'
        head_name = f"head_{head_index}"
        
        # 執行查詢
        search_result = self.client.query_points(
            collection_name = self.collection_name,
            query = query_vector,
            using = head_name,
            limit = top_k,  # 返回最相似的 top_k 筆資料
            with_vectors = True,
            with_payload = True
        )
        return search_result
    
# import torch
# collection_name = "qdrant_msmarcoV2.1_Proposed_1000"
# Qdrant_DB = Qdrant(collection_name)
# random_tensor = torch.randn(1, 32)
# query_vector = random_tensor[0].tolist()
# search_result = Qdrant_DB.query_by_head(1, query_vector, 5)
# print(search_result)