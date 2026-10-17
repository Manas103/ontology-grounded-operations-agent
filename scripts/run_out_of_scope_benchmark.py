"""The 120 out-of-scope adversarial request benchmark.

    PYTHONPATH=. python scripts/run_out_of_scope_benchmark.py

No clinical data load is required; `ScopeGuard` is a stateless membership
check against the 11 read tool names and the 2 write action type names,
run before any tool or action is ever reached.
"""
from __future__ import annotations

from ontology_agent.clinical_scope_guard import KNOWN_IDENTIFIERS, ScopeGuard
from ontology_agent.clinical_scope_requests import (
    OUT_OF_SCOPE_REQUEST_COUNT,
    build_out_of_scope_requests,
)


def main() -> None:
    guard = ScopeGuard()
    requests = build_out_of_scope_requests()
    assert len(requests) == OUT_OF_SCOPE_REQUEST_COUNT

    refused = 0
    wrongly_allowed = 0
    for name in requests:
        result = guard.check(name)
        if result.in_scope:
            wrongly_allowed += 1
            print(f"  UNEXPECTED: {name!r} was treated as in-scope ({result.kind})")
        else:
            refused += 1

    # Sanity check in the other direction: every real catalog entry must
    # still be accepted, so the guard is not simply refusing everything.
    real_names_accepted = sum(1 for name in KNOWN_IDENTIFIERS if guard.check(name).in_scope)

    print(f"out-of-scope requests generated: {len(requests)}")
    print(f"refused before any execution: {refused}")
    print(f"wrongly allowed: {wrongly_allowed}")
    print(f"OUT-OF-SCOPE REFUSAL RATE: {100.0 * refused / len(requests):.4f}%")
    print(
        f"\nreal catalog entries (11 read tools + 2 write actions) still accepted: "
        f"{real_names_accepted} / {len(KNOWN_IDENTIFIERS)}"
    )
    print(f"\nOUT-OF-SCOPE SUMMARY: {refused} of {len(requests)} out-of-scope tool requests refused")


if __name__ == "__main__":
    main()
