// Calls (with summary) for an issue. params: issue, limit
MATCH (c:Call)-[:HAS_ISSUE]->(:Issue {name: $issue})
RETURN c.call_id AS call_id, c.summary AS summary
LIMIT $limit
