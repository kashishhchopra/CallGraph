// Agent actions taken in calls that started in one emotion and ended in another. params: start_emotion, end_emotion, limit
MATCH (c:Call)-[:STARTED_WITH]->(:Emotion {name: $start_emotion}),
      (c)-[:ENDED_IN]->(:Emotion {name: $end_emotion}),
      (c)-[:TOOK_ACTION]->(a:Action)
RETURN a.name AS action, count(DISTINCT c) AS calls
ORDER BY calls DESC LIMIT $limit
