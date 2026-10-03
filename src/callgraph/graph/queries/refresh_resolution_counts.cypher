// Recompute RESOLVED_BY(count). params: outcome
MATCH (i:Issue)<-[:HAS_ISSUE]-(c:Call)-[:ENDED_WITH]->(:Outcome {type: $outcome}), (c)-[:TOOK_ACTION]->(a:Action)
WITH i, a, count(DISTINCT c) AS n
MERGE (i)-[r:RESOLVED_BY]->(a)
SET r.count = n
