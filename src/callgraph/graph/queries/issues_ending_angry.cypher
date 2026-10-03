// Issues whose calls most often end with an angry customer. params: emotion, limit
MATCH (c:Call)-[:HAS_ISSUE]->(i:Issue), (c)-[:ENDED_IN]->(e:Emotion {name: $emotion})
RETURN i.name AS issue, count(c) AS angry_calls
ORDER BY angry_calls DESC LIMIT $limit
