# graphdb.py


from typing import Dict, List

import ladybug
from ladybug import QueryResult


class LadybugGraphDB:
	def __init__(self, db_path: str):
		# Initialize database and connection.
		self.db = ladybug.Database(db_path)
		self.conn = ladybug.Connection(self.db)

		# Initialize table.
		try:
			self.conn.execute(
				"CREATE NODE TABLE Proposition(id STRING, text STRING, PRIMARY KEY (id))"
			)
			self.conn.execute(
				"CREATE NODE TABLE Entity(name STRING, PRIMARY KEY (name))"
			)
			self.conn.execute(
				"CREATE REL TABLE Mentions(FROM Proposition TO Entity)"
			)
		except Exception as e: 
			print(f"Failed to create tables. Exception raised: {e}")


	def add_proposition(self, id: str, text: str) -> None:
		self.conn.execute(
			"CREATE (p:Proposition {id: $id, text: $txt})", 
			{"id": id, "txt": text}
		)


	def add_edge(self, id: str, entity: str) -> None:
		self.conn.execute(
			"MERGE (e:Entity {name: $name})", 
			{"name": entity}
		)
		self.conn.execute(
			"MATCH (p:Proposition), (e:Entity) "
			"WHERE p.id = $pid AND e.name = $ename "
			"CREATE (p)-[:Mentions]->(e)", 
			{"pid": id, "ename": entity}
		)

	
	def batch_write_to_graph(self, propositions: List[Dict[str, str | List[float | int]]]) -> None:
		if not propositions:
			return

		# 1. Create all Propositions in one go.
		prop_list = [
			{"id": prop["id"], "txt": prop["text"]} 
			for prop in propositions
		]
		self.conn.execute(
			"""
			UNWIND $data AS row
			MERGE (p:Proposition {id: row.id})
			SET p.text = row.txt
			""",
			{"data": prop_list}
		)

		# 2. Collect and DEDUPLICATE all edges to be created.
		edge_list = []
		seen_edges = set()
		
		for prop in propositions:
			entities = prop.get("entities", [])
			for entity in entities:
				# Use a tuple to track uniqueness and avoid duplicates 
				# in the same batch.
				edge_tuple = (prop["id"], entity)
				if edge_tuple not in seen_edges:
					seen_edges.add(edge_tuple)
					edge_list.append({"pid": prop["id"], "ename": entity})

		if not edge_list:
			return
		
		# Create all entities first (ensure they exist before linking).
		unique_entities = [
			{"ename": e} 
			for e in list(set(item["ename"] for item in edge_list))
		]
		self.conn.execute(
			"UNWIND $data AS row MERGE (e:Entity {name: row.ename})",
			{"data": unique_entities}
		)
		
		# 3. Use CREATE instead of MERGE to bypass the 
		# unordered_map::at bug.
		query = """
			UNWIND $data AS row
			MATCH (p:Proposition {id: row.pid})
			MATCH (e:Entity {name: row.ename})
			CREATE (p)-[:Mentions]->(e)
		"""
		self.conn.execute(query, {"data": edge_list})


	def multi_hop_expansion(self, id: str, hops: int = 1) -> List[QueryResult]:
		query = f"""
			MATCH (p1:Proposition {{id: $id}})-[:Mentions*1..{hops}]-(p2:Proposition)
			WHERE p1.id <> p2.id
			RETURN DISTINCT p2.text
			LIMIT 10
        """
		return self.conn.execute(query, {"id": id})


	def checkpoint(self) -> None:
		self.conn.execute("CHECKPOINT;")

	
	def close_db(self) -> None:
		self.conn.close()