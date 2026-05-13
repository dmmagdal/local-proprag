# quickstart.py


import json
import os
import random
import shutil

import datasets
from tqdm import tqdm
import pyarrow as pa

from local_vectors import detect_device
from local_proprag import PropRAG


SEED = 1234
random.seed(SEED)

def main():
	# Load the dataset.
	target_dataset = "illuin-conteb/narrative-qa"
	cache_dir = f"./{target_dataset.replace('/', '_')}_cache"
	save_dir = f"./{target_dataset.replace('/', '_')}"
	subsets = ["documents", "queries"]

	# Download the dataset if it's not already available.
	if not os.path.exists(save_dir) or len(os.listdir(save_dir)) == 0:
		for subset in subsets:
			data = datasets.load_dataset(
				target_dataset,
				subset,
				cache_dir=cache_dir,
			)
			data.save_to_disk(os.path.join(save_dir, subset))

		# Clear the cache directory.
		shutil.rmtree(cache_dir)

	# Load the documents and queries.
	documents = datasets.load_from_disk(os.path.join(save_dir, "documents"))
	queries = datasets.load_from_disk(os.path.join(save_dir, "queries"))

	# Load the configuration information.
	with open("config.json", "r") as f:
		config = json.load(f)['proprag']

	# Unpack and organized theh configuration data for each component 
	# of the proprag.
	vector_config = config["vector"]
	graph_config = config["graph"]
	llm_config = config["llm"]

	# Clear any existing tables or databases.
	storage_artifacts = [
		vector_config["vector_db"],
		graph_config["graph_db"],
		"./graph",
		"graph.wal",
	]
	for artifact in storage_artifacts:
		if os.path.exists(artifact):
			if os.path.isdir(artifact):
				shutil.rmtree(artifact)
			else:
				os.remove(artifact)

	# Detect GPU accelerators.
	# device = detect_device(force_cpu=True)
	device = detect_device()
	# nomic-ai/nomic-embed-text-v1.5
	# sentence-transformers/all-MiniLM-L6-v2

	# Initialize proprag with the configuration.
	proprag = PropRAG(
		embed_model_id=vector_config["model_id"],
		vector_db_path=vector_config["vector_db"],
		graph_db_path=graph_config["graph_db"],
		llm_model=llm_config["model_id"],
		token_overlap=vector_config["token_overlap"],
		batch_size=vector_config["batch_size"],
		device=device,
		use_binary=vector_config["use_binary"],
		query_metric=vector_config["metric"],
		model_save_root=vector_config["model_save_root"],
		host=llm_config["host"],  
		gliner_model=llm_config["gliner_model"],
		spacy_model=llm_config["spacy_model"],
	)

	# Define schema (this is heavily dependent upon the datasets) and
	# pass that to the proprag so that the vectordb can build the 
	# table.
	schema = pa.schema([
		pa.field("entities", pa.list_(pa.string())),
		pa.field("vector", pa.list_(
			pa.uint8() if vector_config["use_binary"] else pa.float32(), 
			proprag.get_dims()
		)),
		pa.field("id", pa.string()),
		pa.field("text", pa.string())
	])
	proprag.build_vector_table(
		table_name=vector_config["table_name"],
		schema=schema,
	)

	# Ingest and index the documents to the proprag.
	for split_name, data in documents.items():
		# for idx in tqdm(data, desc=f"Ingesting {split_name} split into Proposition RAG"):
		# 	proprag.ingest(
		# 		text=doc["chunk"], 
		# 		doc_id=doc["chunk_id"],
		# 		table_name=vector_config["table_name"]
		# 	) # Single-document ingestion.
		for idx in tqdm(range(0, len(data), vector_config["batch_size"]), desc=f"Ingesting {split_name} split into Proposition RAG"):
			docs = data[idx:idx + vector_config["batch_size"]]
			doc_list = [
				{"id": chunk_id, "text": chunk}
				for chunk_id, chunk in zip(docs["chunk_id"], docs["chunk"])
			]
			proprag.batch_ingest(
				documents=doc_list,
				table_name=vector_config["table_name"]
			) # Batch document ingeston.

	# Perform a query on the proprag.
	sampled_queries = queries.shuffle(seed=SEED).select(range(5))
	for query in sampled_queries:
		question, chunk_id, answer = query["og_query"], query["chunk_id"], query["answer"]
		proprag_answer = proprag.query(question)

		# Output results.
		print(f"Question: {question}")
		print(f"Expected answer: {answer} (chunk id {chunk_id})")
		print(f"Generated answer: {proprag_answer}")
		print("-" * 72)

	# Clear all tables or databases since we're done.
	for artifact in storage_artifacts:
		if os.path.exists(artifact):
			if os.path.isdir(artifact):
				shutil.rmtree(artifact)
			else:
				os.remove(artifact)

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()