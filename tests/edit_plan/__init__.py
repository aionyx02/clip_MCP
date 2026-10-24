"""Tests for the edit plan: what it refuses, and what it compiles to.

The compiler is the deterministic half of the design, so most of these are
arithmetic: given this plan and this footage, these are the seconds. The rest
check that a plan which does not hold up is handed back rather than quietly
repaired, because a compiler that fixes things is making decisions it was
built to stay out of.
"""
