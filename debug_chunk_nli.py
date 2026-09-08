"""
Debug script: fetches a specific chunk's FULL text from the DB and shows
exactly what sentence(s) get selected for NLI, and what the model scores
them as. Run with:  python debug_chunk_nli.py <chunk_id> <claim text>

Example:
  python debug_chunk_nli.py 136 "ZJIT lifts local variables into SSA values"
"""

import sys
sys.path.insert(0, ".")

from backend.database.db import SessionLocal
from backend.database.models import Chunk
from backend.nlp.sentence_selection import select_relevant_sentences, split_sentences
from backend.nlp.nli import get_nli_model, resolve_self_references

if len(sys.argv) < 3:
    print("Usage: python debug_chunk_nli.py <chunk_id> <claim text>")
    sys.exit(1)

chunk_id = int(sys.argv[1])
claim = sys.argv[2]

db = SessionLocal()
chunk = db.query(Chunk).filter(Chunk.id == chunk_id).first()
db.close()

if not chunk:
    print(f"No chunk found with id {chunk_id}")
    sys.exit(1)

print("=" * 70)
print(f"FULL CHUNK TEXT (chunk_id={chunk_id}):")
print("=" * 70)
print(chunk.text)
print()

print("=" * 70)
print("SENTENCES (as split by our splitter):")
print("=" * 70)
sentences = split_sentences(chunk.text)
for i, s in enumerate(sentences):
    print(f"[{i}] {s}")
print()

print("=" * 70)
print(f"CLAIM: {claim}")
print("=" * 70)
selected_text, debug_info = select_relevant_sentences(chunk.text, claim, top_n=2, return_debug=True)
print()
print("Sentences ranked by similarity to claim:")
if "all_sentences_ranked" in debug_info:
    for item in debug_info["all_sentences_ranked"]:
        print(f"  sim={item['similarity']:.4f}  {item['sentence']}")
print()
print(f"SELECTED PREMISE (what gets sent to NLI): {selected_text!r}")
print()

# Now run it through NLI with and without subject_name resolution
subject_name = "ZJIT"
resolved = resolve_self_references(selected_text, subject_name)
print(f"AFTER 'we/our' -> '{subject_name}' substitution: {resolved!r}")
print()

model = get_nli_model()
scores = model.predict([(resolved, claim)])[0]
print(f"Raw NLI scores [contradiction, entailment, neutral]: {scores}")
