// Top call reasons. params: limit
MATCH (c:Call)-[:HAS_INTENT]->(i:Intent)
RETURN i.name AS intent, count(c) AS calls
ORDER BY calls DESC LIMIT $limit
