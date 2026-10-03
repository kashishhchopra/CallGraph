// Calls with a given intent where a rubric item failed. params: intent, rubric_item, limit
MATCH (c:Call)-[:HAS_INTENT]->(:Intent {name: $intent}),
      (c)-[:HAS_CHECK]->(k:ComplianceCheck {rubric_item: $rubric_item, passed: false})
RETURN c.call_id AS call_id, k.evidence AS evidence
LIMIT $limit
