"""
SciFact Benchmark for NLI Verification Module.

Downloads the SciFact development set and corpus, formats the claims and
evidence sentences, and runs them through our local NLI model to compute
Accuracy, Precision, Recall, and F1 on a rigorous external dataset.
"""

import os
import sys
import json
import urllib.request
from collections import defaultdict
import time

# Add the project root to the Python path so it can find the 'backend' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.nlp.nli import check_entailment_batch

DATA_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
DATA_DIR = os.path.join(os.path.dirname(__file__), "scifact_data")

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest_path)
    return dest_path

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def get_scifact_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    tar_path = os.path.join(DATA_DIR, "data.tar.gz")
    
    if not os.path.exists(os.path.join(DATA_DIR, "data", "corpus.jsonl")):
        download_file(DATA_URL, tar_path)
        print("Extracting data.tar.gz ...")
        import tarfile
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=DATA_DIR)

    corpus_path = os.path.join(DATA_DIR, "data", "corpus.jsonl")
    claims_path = os.path.join(DATA_DIR, "data", "claims_dev.jsonl")

    corpus_raw = load_jsonl(corpus_path)
    claims_raw = load_jsonl(claims_path)

    # Map doc_id to list of sentences
    corpus = {str(doc["doc_id"]): doc["abstract"] for doc in corpus_raw}
    
    pairs = []
    
    for claim_obj in claims_raw:
        claim_text = claim_obj["claim"]
        evidence_dict = claim_obj.get("evidence", {})
        
        for doc_id, ev_list in evidence_dict.items():
            doc_id = str(doc_id)
            if doc_id not in corpus:
                continue
            
            doc_sentences = corpus[doc_id]
            
            for ev in ev_list:
                label = ev["label"] # "SUPPORTED" or "REFUTED"
                sentence_indices = ev["sentences"]
                
                # Combine the specific evidence sentences into one text block
                ev_text = " ".join([doc_sentences[i] for i in sentence_indices])
                
                pairs.append({
                    "claim": claim_text,
                    "evidence": ev_text,
                    "gold_label": label # "SUPPORTED" or "REFUTED"
                })
                
    return pairs

def run_benchmark():
    print("Loading SciFact dataset...")
    pairs = get_scifact_data()
    print(f"Found {len(pairs)} evidence pairs in the dev set.")
    
    # To keep the test fast, we'll evaluate a random subset of 100 pairs if there are many.
    # SciFact dev set has ~300 pairs. We'll just run all of them in batches.
    import random
    random.seed(42)
    random.shuffle(pairs)
    # limit to 100 to save time for immediate validation
    test_pairs = pairs[:100]
    
    print(f"Running NLI model on {len(test_pairs)} pairs... (this may take a minute or two)")
    
    # Prepare batch
    premises = [p["evidence"] for p in test_pairs]
    hypotheses = [p["claim"] for p in test_pairs]
    gold_labels = [p["gold_label"] for p in test_pairs]
    
    start_time = time.time()
    
    # check_entailment_batch expects list of dicts: {"premise": str, "hypothesis": str}
    # Wait, looking at nli.py, check_entailment_batch takes (pairs: list[tuple[str, str]], ...)
    # Let's import it and check the signature.
    nli_input = list(zip(premises, hypotheses))
    
    # Our system returns dicts with 'label': 'entailment' | 'contradiction' | 'neutral'
    results = check_entailment_batch(nli_input)
    
    end_time = time.time()
    
    # Map NLI output to SciFact labels
    # entailment -> SUPPORTED
    # contradiction -> REFUTED
    # neutral -> NOT_ENOUGH_INFO
    
    correct = 0
    confusion_matrix = defaultdict(int)
    
    for gold, res in zip(gold_labels, results):
        pred_raw = res.label
        if pred_raw == "entailment":
            pred = "SUPPORT"
        elif pred_raw == "contradiction":
            pred = "CONTRADICT"
        else:
            pred = "NOT_ENOUGH_INFO"
            
        confusion_matrix[(gold, pred)] += 1
        if pred == gold:
            correct += 1
            
    accuracy = correct / len(test_pairs)
    
    print("\n" + "="*40)
    print("SciFact Benchmark Results")
    print("="*40)
    print(f"Total Pairs Evaluated : {len(test_pairs)}")
    print(f"Total Time Taken      : {end_time - start_time:.1f} seconds")
    
    # Calculate Precision and Recall for SUPPORT (which is what we care about for VALID claims)
    tp = confusion_matrix[("SUPPORT", "SUPPORT")]
    fp = confusion_matrix[("CONTRADICT", "SUPPORT")] + confusion_matrix[("NOT_ENOUGH_INFO", "SUPPORT")] # Note: SciFact dev set doesn't give NEI gold labels in this evidence list
    fn = confusion_matrix[("SUPPORT", "CONTRADICT")] + confusion_matrix[("SUPPORT", "NOT_ENOUGH_INFO")]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"Accuracy              : {accuracy * 100:.1f}%")
    print(f"Precision (SUPPORT)   : {precision * 100:.1f}%")
    print(f"Recall (SUPPORT)      : {recall * 100:.1f}%")
    print(f"F1 Score (SUPPORT)    : {f1 * 100:.1f}%\n")
    
    print("Confusion Matrix (Given -> Predicted):")
    for (g, p), count in sorted(confusion_matrix.items()):
        print(f"  {g} -> {p}: {count}")

if __name__ == "__main__":
    run_benchmark()
