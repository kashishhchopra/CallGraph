// Remove everything derived from one call before rewriting it (keeps re-processing idempotent). params: call_id
MATCH (c:Call {call_id: $call_id})
OPTIONAL MATCH (c)-[:HAS_TURN]->(t:Turn)
OPTIONAL MATCH (c)-[:HAS_CHECK]->(k:ComplianceCheck)
OPTIONAL MATCH (c)-[:HAS_PARTICIPANT]->(s:Speaker)
OPTIONAL MATCH (c)-[r:HAS_INTENT|HAS_ISSUE|TOOK_ACTION|ENDED_WITH|STARTED_WITH|ENDED_IN]->()
DELETE r
WITH c, collect(DISTINCT t) AS ts, collect(DISTINCT k) AS ks, collect(DISTINCT s) AS ss
FOREACH (x IN ts | DETACH DELETE x)
FOREACH (x IN ks | DETACH DELETE x)
FOREACH (x IN ss | DETACH DELETE x)
