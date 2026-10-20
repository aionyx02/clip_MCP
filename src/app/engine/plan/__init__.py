"""Turning an edit plan into timeline operations, and checking one before that happens.

This is the deterministic half of the split the whole design rests on. A model
decides which footage is used and what it is for; everything from there — where
each cut lands, how much air it gets, what gets merged, how long the result
runs — happens here, as a pure function of the plan and the semantic timeline.

Two things follow. The same plan always compiles to the same cut, so an edit
can be reproduced and two plans can be compared rather than two videos. And
every rule in here is a unit test rather than a sentence in a guide hoping to
be followed.

Nothing is repaired on the way through. A plan that does not hold up is handed
back with the reasons, because a compiler that quietly fixed things would be
making the decisions it was built to stay out of.
"""
