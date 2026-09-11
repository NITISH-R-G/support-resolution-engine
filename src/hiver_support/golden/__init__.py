"""The golden evaluation set: candidates, human annotations, and the wall between them.

Protocol: ``docs/GOLDEN_SET.md``, frozen before any example was drawn from the test pool.

The package is split so that the thing a human reads during annotation and the thing that
carries a label are different objects in different files. A ``GoldenCandidate`` has no field
for a label and no field for the brand's reply; a ``GoldenAnnotation`` cannot exist without a
named human annotator. Neither arrangement is a convention that could be forgotten — both
raise.
"""
