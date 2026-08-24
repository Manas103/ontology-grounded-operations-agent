"""Ontology-grounded operations agent.

A typed object model for field equipment maintenance operations
(sites, technicians, assets, parts, work orders), a read-only
question-answering path that may only touch that typed model through
a fixed catalog of parameterized tool functions, and an action-proposal
path where every state change is a JSON-Schema-validated action a human
must approve before anything is applied.
"""
