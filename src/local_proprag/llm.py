# llm.py
# This file contains functions for interacting with LLMs via Ollama.


import requests


class LLM:
	def __init__(
		self, 
		llm_model: str,
	):
		self.LLM_MODEL = llm_model


	def call_llm(self, prompt: str) -> str:
		return ""
	

	def generate_response(self, prompt: str) -> str:
		return ""


class OllamaLLM(LLM):
	def __init__(
		self, 
		llm_model: str,
		host: str = "http://localhost:11434", 
	):
		self.LLM_MODEL = llm_model
		self.OLLAMA_URL = host.rstrip("/")


	def call_llm(self, prompt: str, format: str = "") -> str:
		payload = {
			"model": self.LLM_MODEL, 
			"prompt": prompt, 
			"stream": False, 
		}
		if format != "":
			payload["format"] = format

		res = requests.post(
			f"{self.OLLAMA_URL}/api/generate", 
			json=payload
		)
		res.raise_for_status()
		return res.json()['response']
		

	def generate_response(self, prompt: str) -> str:
		return self.call_llm(prompt)