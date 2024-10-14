from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping, Optional
import requests
import json
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

class ArgoChatModel(LLM):

    model_type: str = "gpt4"
    url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
    temperature: Optional[float] = 0.1
    system: str = "You are a large language model with the name Argo."
    top_p: Optional[float]= 0.1
    user: str = "cels"

    @property
    def _llm_type(self) -> str:
        return "ArgoLLM"
    
    def _call(
        self,
        prompts: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:


        headers = {
            "Content-Type": "application/json"
        }

        params = {
            **self._get_model_default_parameters,
            **kwargs,
            "prompt": [prompts],
            "stop": []
        }

        params_json = json.dumps(params);
        response = requests.post(self.url, headers=headers, data=params_json)
 
        if response.status_code == 200:
            parsed = json.loads(response.text)
            return parsed['response']
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")
 
    @property
    def _get_model_default_parameters(self):
        return {
            "user": self.user,
            "model": self.model_type,
            "system": "" if self.system is None else self.system,
            "temperature": self.temperature,
            "top_p":  self.top_p
        }
 
    @property
    def model(self):
        return self.model_type
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}
 
    @property
    def _generations(self):
        return
    
class ArgoEmbeddingWrapper:
    url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/"
    user: str = "cels"  # Replace with dynamic user handling or environment variable

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    def _call_argo(self, texts: List[str]) -> List[List[float]]:
        """
        Calls the Argo embedding API to get embeddings for the input texts.
        Args:
            texts: List of texts to embed.
        
        Returns:
            List of embeddings, where each embedding is a list of floats (vectors).
        """
        # Prepare the payload
        data = {
            "user": self.user,
            "model": self.model,  # Embedding model name, adjust based on Argo API
            "prompt": texts  # The texts to embed
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(self.url, data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            # Parse and return embeddings
            parsed = response.json()
            return parsed.get("embeddings", [])  # Adjust this according to Argo's response structure
        else:
            raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents (texts).
        Args:
            texts: List of documents to embed.
        
        Returns:
            List of embeddings (vectors).
        """
        return self._call_argo(texts)

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query.
        Args:
            query: The query to embed.
        
        Returns:
            A single embedding vector.
        """
        embeddings = self._call_argo([query])
        return embeddings[0] if embeddings else []