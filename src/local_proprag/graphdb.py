# graphdb.py


from typing import Dict, List, Tuple

import ladybug
from ladybug import QueryResult


class LadybugGraphDB:
	def __init__(self, db_path: str, hipporagv2: bool = False):
		# Initialize database and connection.
		self.db = ladybug.Database(db_path)
		self.conn = ladybug.Connection(self.db)
		self.hipporag2 = hipporagv2

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

	
	def batch_write_to_graph(self, propositions: List[Dict[str, str | List[float | int]]]):
		# 1. Create all Propositions in one go (if supported)
		# 2. Collect all edges to be created
		edge_list = []
		for prop in propositions:
			for entity in prop["entities"]:
				edge_list.append({"pid": prop["id"], "ename": entity})
		
		# 3. Use a single MATCH/CREATE pattern or a UNWIND statement
		# Example Cypher for batching:
		query = """
		UNWIND $data AS row
		MERGE (e:Entity {name: row.ename})
		WITH row, e
		MATCH (p:Proposition {id: row.pid})
		CREATE (p)-[:Mentions]->(e)
		"""
		self.conn.execute(query, {"data": edge_list})


	def multi_hop_expansion(self, id: str, hops: int = 1) -> List[str]:
		query = f"""
			MATCH (p1:Proposition {{id: $id}})-[:Mentions*1..{hops}]-(p2:Proposition)
			WHERE p1.id <> p2.id
			RETURN DISTINCT p2.text
			LIMIT 10
        """
		return self.conn.execute(query, {"id": id})


	def checkpoint(self) -> None:
		self.conn.execute("PRAGMA wal_checkpoint(FULL);")