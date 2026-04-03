from abc import ABC,abstractmethod
import logging
from openai import OpenAI
from typing import Union, Optional

logger = logging.getLogger(__name__)
class BaseChat(ABC):
    @abstractmethod
    def ask(self, prompt: str) -> str:
        pass

class BaseEmbedding(ABC):
    @abstractmethod
    def encode(self, text: str) -> list:
        pass

class Embedding(BaseEmbedding):
    def __init__(self,api_key: str , api_base: str, model: str):
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    def encode(self, sentences: Union[str,list]) -> list:
        num=0
        while(num<5):
            if(isinstance(sentences,str)):
                completion = self.client.embeddings.create(
                    input=sentences,
                    model=self.model,
                )
                try:
                    return completion.data[0].embedding
                except:
                    num+=1
            elif(isinstance(sentences,list)):
                completion = self.client.embeddings.create(
                    input=sentences,
                    model=self.model,
                )
                try:
                    return [c.embedding for c in completion.data]
                except:
                    num+=1
        if(num>=5):
            logger.info(f"try {num} times but can not connected")
            return None
    
class Chat(BaseChat):
    def __init__(self, api_key: str, api_base: str, model:str,**kwargs):
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
        self.model=model
        self.top_p = kwargs.get("top_p",0.8)
        self.temperature =kwargs.get("temperature",0.5)
        self.tools=kwargs.get("tools",None)
        
    def ask(self, prompt: str) -> str:
        num=0
        while(num<5):
            completion = self.client.chat.completions.create(
            # model="yi-lightning","yi-medium",
                model=self.model,
                messages=[{"role": "user", "content": f"{prompt}"}],
                # top_p=self.top_p,
                temperature=self.temperature,
                tools=self.tools,
                stream=False,
            )
            try:
                return completion.choices[0].message.content
            except:
                num+=1
        if(num>=5):
            logger.info(f"try {num} times but can not connected")
            return None