# proprag.py


import gc
import json
from pathlib import Path
from typing import List

from local_vectors import LocalEmbedder, LanceDBConnection
import networkx as nx
import pandas as pd
import pyarrow as pa

from .extractor import PropositionExtractor
from .graphdb import LadybugGraphDB
from .llm import OllamaLLM


class PropRAG:
	def __init__(self, 
		embed_model_id: str,
		vector_db_path: str,
		graph_db_path: str,
		gliner_model: str,
		spacy_model: str,
		llm_model: str,
		entity_items: List[str] = None,												# PropositionExtrator kwargs
		token_overlap: int = 128,													# Embedder kwargs
		batch_size: int = 8,
		device: str = "cpu",
		use_binary: bool = False,
		query_metric: str = "cosine",
		model_save_root: str = Path.home() / ".cache" / "local-graphrag" / "models",
		host: str = "http://localhost:11434",   									# LLM kwargs
	):
		# Initialize proposition extractor.
		self.prop_extractor = PropositionExtractor(
			gliner_model=gliner_model,
			spacy_model=spacy_model,
			entity_items=entity_items,
			device=device,
			model_save_root=model_save_root
		)
		
		# Initialize the text embedder.
		self.embedder = LocalEmbedder(
			model_id=embed_model_id,
			model_save_root=model_save_root,
			token_overlap=token_overlap,
			batch_size=batch_size,
			device=device,
		)

		# Initialize the vectordb.
		self.vectordb = LanceDBConnection(
			vector_db_path
		)

		# Flag whether we're using binary embeddings or full precision 
		# (as per supported on local-vectors).
		self.use_binary = use_binary

		# Initialize the graphdb.
		self.graphdb = LadybugGraphDB(
			db_path=graph_db_path,
		)

		# Initialize LLM.
		self.llm = OllamaLLM(
			llm_model=llm_model,
			host=host,
		)

		# Set the query metric.
		self.metric = query_metric


	def get_dims(self) -> int:
		model_metadata = self.embedder.model_metadata
		return model_metadata["binary_dims"] if self.use_binary else model_metadata["dims"]


	def build_vector_table(self, table_name: str, schema: pa.Schema) -> None:
		self.vectordb.create_table(table_name, schema)


	def set_query_metric(self, query_metric: str) -> None:
		self.metric = query_metric


	def ingest(self, text: str, doc_id: str, table_name: str) -> None:
		# Error checking in case the user hasn't initialized the 
		# desired table yet.
		if table_name not in self.vectordb.table_names():
			raise ValueError(f"Table {table_name} has not yet been initialize for the vectordb. Current tables include {', '.join(self.vectordb.table_names())}")

		# Extract the propositions from the text.
		props = self.prop_extractor.extract_propositions(doc_id, text)
		for prop in props:
			# Compute the vector embeddings.
			vectors = self.embedder.embed_text(
				prop["text"],
				truncate=True,
				to_binary=self.use_binary,
				vectors_only=True
			)[0]
			prop["vector"] = vectors["vector_binary"] if self.use_binary else vectors["vector_full"]

			# Update the graph.
			self.graphdb.add_proposition(prop["id"], prop["text"])
			for entity in prop["entities"]:
				# self.graphdb.add_edge(prop["id"], entity)
				self.graphdb.add_edge(prop["id"], entity)

		# Update the vector database.
		self.vectordb.update_table(table_name, props)
		gc.collect()


	def query(self, query: str, table_name: str, top_k: int = 5, hops: int = 1) -> str:
		# Error checking in case the user hasn't initialized the 
		# desired table yet.
		if table_name not in self.vectordb.table_names():
			raise ValueError(f"Table {table_name} has not yet been initialize for the vectordb. Current tables include {', '.join(self.vectordb.table_names())}")
		
		# Semantic search (via vector database).
		query_vector = self.embedder.embed_text(
			self.query,
			truncate=True,
			to_binary=self.use_binary,
			vectors_only=True
		)[0]
		query_vector = query_vector["vector_binary"] if self.use_binary else query_vector["vector_full"]
		results = pd.DataFrame(
			self.vectordb.search_table(
				table_name, query_vector, top_k=top_k, metric=self.metric
			)
		)

		# Propositional expansion from the graph database.
		contextual_ids = results["id"].tolist()
		expanded_context = set(results["text"].tolist())

		# Cypher multi-hop expansion.
		for passage_id in contextual_ids:
			results = self.graphdb.multi_hop_expansion(
				passage_id, hops=hops
			)
			while results.has_next():
				expanded_context.add(results.get_next()[0])

		# Response synthesis.
		final_context = "\n".join(list(expanded_context))
		prompt = f"""
			Answer the query using the provided propositions.
			Context:
			{final_context}
			
			Query: {query}
		"""
		return self.llm.generate_response(prompt)

