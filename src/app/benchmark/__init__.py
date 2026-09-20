"""Scoring a cut without rendering it.

Three layers are planned. L1 is the test suite, which already says whether the
arithmetic is right. L3 needs a model to judge a finished video and is not here
yet. This is L2: given the footage, what was asked for, and what a plan chose,
work out whether the choice broke any of the rules a cut is not allowed to
break.

L2 is the layer worth having first because it is cheap. A plan is structured
data, so most of what matters can be measured rather than watched, and nothing
has to be encoded to measure it. That makes it fast enough to run on every
commit, which is the only way a score stops a regression rather than
documenting one.
"""
