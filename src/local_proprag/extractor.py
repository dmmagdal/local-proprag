# extractor.py


import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Dict, List

from gliner import GLiNER
import requests
import spacy
from spacy.language import Language


class PropositionExtractor:
	def __init__(self, 
		gliner_model: str, 
		spacy_model: str, 
		entity_items: List[str] = None,
		device: str = "cpu",
		model_save_root: str = Path.home() / "local_lightrag" / "models",
	):
		self.gliner_model = gliner_model
		self.spacy_model = spacy_model
		self.model_save_root = model_save_root
		self.device = device

		# Initialize/load models.
		self.ner = self._load_gliner_model()
		self.nlp = self._load_spacy_model()

		default_items = [
			"Person", "Org", "Product", "Event", "Concept"
		]
		self.entity_items = entity_items if entity_items is not None else default_items
		

	def extract_propositions(self, doc_id: str, document: str) -> List[Dict[str, str]]:
		doc = self.nlp(document)
		propositions = []

		for idx, sent in enumerate(doc.sents):
			# Extract entities.
			entities = self.ner.predict_entities(
				sent.text, self.entity_items
			)

			# Create a proposition for every sentence, tagged with its 
			# entities.
			entity_list = [e['text'].lower() for e in entities]

			propositions.append({
				"id": f"{doc_id}_{idx}",
				"text": sent.text.strip(),
				"entities": entity_list,
			})

		return propositions

		
	def _load_gliner_model(self) -> GLiNER:
		model_path = str(Path(self.model_save_root) / self.gliner_model.replace("/", "_"))

		# Check for path and that path is a directory.
		if not os.path.exists(model_path) or not os.path.isdir(model_path):
			os.makedirs(model_path, exist_ok=True)

		# Check for path to be populated with files.
		if len(os.listdir(model_path)) == 0:
			print(f"Model {self.gliner_model} needs to be downloaded.")

			# Connectivity check.
			try:
				response = requests.get("https://huggingface.co/", timeout=5)
				if response.status_code != 200:
					raise ConnectionError
			except Exception:
				print(f"Unable to reach Hugging Face to download {self.gliner_model}.")
				exit(1)

			# Create temporary cache path
			cache_path = model_path + "_tmp"
			os.makedirs(cache_path, exist_ok=True)

			# 1. Download/Load model into temporary cache. GLiNER uses 
			# HF's cache_dir internally.
			model = GLiNER.from_pretrained(
				self.gliner_model,
				cache_dir=cache_path,
				load_tokenizer=True,
			)

			# 2. Save the model to the final destination. This saves 
			# the config, model weights, and tokenizer files.
			model.save_pretrained(model_path)

			# 3. Clean up the temporary cache.
			shutil.rmtree(cache_path)
		
		# 4. Load the model from the local path. Wet 
		# local_files_only=True to ensure it doesn't try to ping HF 
		# again.
		model = GLiNER.from_pretrained(
			model_path,
			local_files_only=True
		)

		# Return the model.
		return model
	

	def _load_spacy_model(self) -> Language:
		try:
			# Attempt to load the model.
			return spacy.load(self.spacy_model)
		except OSError:
			print(f"Model {self.spacy_model} not found. Downloading...")
			# Use uv to install it dynamically if available, otherwise 
			# fallback to spacy.
			try:
				subprocess.run(
					[
						sys.executable, "-m", "pip", "install", 
						f"https://github.com/explosion/spacy-models/releases/download/{self.spacy_model}-3.7.1/{self.spacy_model}-3.7.1-py3-none-any.whl"
					], 
					check=True
				)
			except:
				# Standard spacy download fallback
				spacy.cli.download(self.spacy_model)
			
			return spacy.load(self.spacy_model)