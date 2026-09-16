"""
Day 16 (Fix 3) diagnostic: sanity-check the new NLI model before trusting
it in the pipeline. Mirrors diagnose_nli.py's approach from Day 9.
Run with:  python diagnose_nli_fix3.py
"""

from backend.nlp.nli import get_nli_model, check_entailment

print("Loading model (first run downloads ~1.7GB, may take a few minutes)...")
loaded = get_nli_model()
print(f"Label map read from model config: {loaded.label_map}")
print()

print("=" * 60)
print("TEST 1: Obvious entailment/contradiction (sanity check)")
print("=" * 60)
tests = [
    ("A man is playing a guitar.", "A man is playing an instrument.", "entailment"),
    ("The sky is blue.", "The sky is green.", "contradiction"),
    ("The cat sat on the mat.", "The cat sat on the mat.", "entailment"),
]
for premise, hyp, expected in tests:
    result = check_entailment(premise, hyp)
    status = "PASS" if result.label == expected else "FAIL"
    print(f"{status}: P='{premise}' H='{hyp}' -> {result.label} ({result.confidence:.2f}) [expected {expected}]")

print()
print("=" * 60)
print("TEST 2: The exact ZJIT case that needed the coreference fix")
print("=" * 60)
premise = "We lift locals to SSA values during SSA construction."
claim = "ZJIT lifts local variables into SSA values"
result_no_fix = check_entailment(premise, claim)
result_with_fix = check_entailment(premise, claim, subject_name="ZJIT")
print(f"Without subject_name fix: {result_no_fix.label} ({result_no_fix.confidence:.2f})")
print(f"With subject_name fix:    {result_with_fix.label} ({result_with_fix.confidence:.2f})")
print()
print("If the new model handles coreference better than the old one, the")
print("'without fix' result might already show entailment - worth noting")
print("if so, since it would mean Fix 3 also resolves the Day 9 issue for free.")
