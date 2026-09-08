"""
Day 9 diagnostic: run this directly to sanity-check the NLI model before
trusting its output in the pipeline.
Run with:  python diagnose_nli.py
"""

from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

print("=" * 60)
print("TEST 1: Obvious entailment, SHORT sentences")
print("=" * 60)
pairs = [
    ("A man is playing a guitar.", "A man is playing an instrument."),
    ("The cat sat on the mat.", "The cat sat on the mat."),  # identical = definitely entailment
    ("The sky is blue.", "The sky is green."),  # contradiction
]
scores = model.predict(pairs)
for (premise, hyp), score in zip(pairs, scores):
    print(f"P: {premise}")
    print(f"H: {hyp}")
    print(f"raw scores: {score}")
    print()

print("=" * 60)
print("TEST 2: Same claim, SHORT vs LONG premise")
print("=" * 60)
short_premise = "We lift locals to SSA values during SSA construction."
long_premise = (
    "not effectively be able to bring locals into SSA. So we lift locals to SSA "
    "values during SSA construction. Below is a sample trace of constructing HIR "
    "from a Ruby method that initializes two locals a and b and then constructs "
    "an array containing both. We use the FrameState structure in our abstract "
    "interpretation of the bytecode to map local variables to SSA values."
)
claim = "ZJIT lifts local variables into SSA values"

for label, premise in [("SHORT", short_premise), ("LONG", long_premise)]:
    score = model.predict([(premise, claim)])[0]
    print(f"{label} premise raw scores: {score}")

print()
print("=" * 60)
print("TEST 3: Does replacing 'we/our' with the system name fix it?")
print("=" * 60)
fixed_premise = short_premise.replace("We lift", "ZJIT lifts")
score = model.predict([(fixed_premise, claim)])[0]
print(f"Fixed premise: {fixed_premise}")
print(f"raw scores: {score}")

print()
print("NOTE: cross-encoder/nli-deberta-v3-base label order should be")
print("[contradiction, entailment, neutral] per its model card.")
print("If TEST 1's identical-sentence pair doesn't show index 1 as clearly")
print("highest, our label order assumption is WRONG.")
print("If TEST 2's SHORT premise scores much more clearly than LONG,")
print("premise length is diluting the signal.")
