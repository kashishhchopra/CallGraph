// How an issue is usually resolved (multi-hop). params: issue, outcome, limit
MATCH (i:Issue {name: $issue})<-[:HAS_ISSUE]-(c:Call)-[:TOOK_ACTION]->(a:Action),
      (c)-[:ENDED_WITH]->(:Outcome {type: $outcome})
RETURN a.name AS action, count(DISTINCT c) AS times
ORDER BY times DESC LIMIT $limit
