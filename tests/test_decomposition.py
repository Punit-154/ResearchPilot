"""
Unit tests for claim decomposition logic.

All tests use mocked LLM responses via unittest.mock.patch — no live API
calls are made. Tests are written against the public interface of
decompose_claims() before the implementation exists, following the same
pattern as the NLI / citation-verification tests in this project.

Run with:  python -m pytest tests/test_decomposition.py -v
"""
import json
import pytest
from unittest.mock import patch, call

from backend.llm.synthesis import Claim

# Module under test — written after these tests are approved
from backend.llm.decomposition import decompose_claims


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

MOCK_TARGET = "backend.llm.decomposition.call_grok"


def _patch_llm(response_str: str):
    """Convenience context manager: patches call_grok to return response_str."""
    return patch(MOCK_TARGET, return_value=response_str)


def _atomic_response(*claim_ids_and_texts):
    """
    Build a mocked decomposition response where every claim is returned
    as-is (already atomic, one sub_claim per parent, text unchanged).

    Usage: _atomic_response(("C01", "Fact one.", ["E01"]))
    """
    decomposed = [
        {
            "parent_id": cid,
            "sub_claims": [{"text": text, "citations": citations}],
        }
        for cid, text, citations in claim_ids_and_texts
    ]
    return json.dumps({"decomposed": decomposed})


# ─────────────────────────────────────────────────────────────────────────────
# Test: edge / guard cases
# ─────────────────────────────────────────────────────────────────────────────

class TestDecomposeClaimsGuards:

    def test_empty_list_returns_empty_without_llm_call(self):
        """No LLM call is made when there are zero claims."""
        with patch(MOCK_TARGET) as mock:
            result = decompose_claims([])
        mock.assert_not_called()
        assert result == []

    def test_single_atomic_claim_returns_unchanged(self):
        """
        A claim the LLM judges as already atomic is returned with the
        SAME id, SAME text, SAME citations, and parent_claim_id=None.
        It must not be artificially split.
        """
        claim = Claim(id="C01", text="ZJIT lifts locals into SSA values.", citation_labels=["E01"])
        mocked = _atomic_response(("C01", "ZJIT lifts locals into SSA values.", ["E01"]))

        with _patch_llm(mocked):
            result = decompose_claims([claim])

        assert len(result) == 1
        r = result[0]
        assert r.id == "C01"
        assert r.text == "ZJIT lifts locals into SSA values."
        assert r.citation_labels == ["E01"]
        assert r.parent_claim_id is None  # atomic → no parent

    def test_llm_called_exactly_once_regardless_of_claim_count(self):
        """Batching: 7 claims should trigger exactly ONE LLM call, not 7."""
        claims = [
            Claim(id=f"C{i:02d}", text=f"Fact {i}.", citation_labels=["E01"])
            for i in range(1, 8)
        ]
        mocked = json.dumps({
            "decomposed": [
                {"parent_id": f"C{i:02d}", "sub_claims": [{"text": f"Fact {i}.", "citations": ["E01"]}]}
                for i in range(1, 8)
            ]
        })

        with patch(MOCK_TARGET) as mock:
            mock.return_value = mocked
            decompose_claims(claims)

        assert mock.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test: decomposition of compound claims
# ─────────────────────────────────────────────────────────────────────────────

class TestDecomposeClaimsDecomposition:

    def test_compound_claim_split_into_two_sub_claims(self):
        """
        A compound claim is split into sub-facts.
        Sub-claims get sequential IDs (<parent>_a, <parent>_b) and
        parent_claim_id set to the original claim id.
        """
        claim = Claim(
            id="C02",
            text="ZJIT uses SSA values for locals, unlike other JITs that use memory loads and stores.",
            citation_labels=["E01", "E03"],
        )

        mocked = json.dumps({
            "decomposed": [
                {
                    "parent_id": "C02",
                    "sub_claims": [
                        {"text": "ZJIT uses SSA values for local variables.", "citations": ["E01"]},
                        {"text": "Other JITs use memory loads and stores for local variables.", "citations": ["E03"]},
                    ],
                }
            ]
        })

        with _patch_llm(mocked):
            result = decompose_claims([claim])

        assert len(result) == 2

        a = result[0]
        assert a.id == "C02_a"
        assert a.text == "ZJIT uses SSA values for local variables."
        assert a.citation_labels == ["E01"]
        assert a.parent_claim_id == "C02"

        b = result[1]
        assert b.id == "C02_b"
        assert b.text == "Other JITs use memory loads and stores for local variables."
        assert b.citation_labels == ["E03"]
        assert b.parent_claim_id == "C02"

    def test_compound_claim_split_into_three_sub_claims(self):
        """Three-way split uses _a, _b, _c suffixes."""
        claim = Claim(
            id="C03",
            text="System A is fast, B is accurate, and C is cheap.",
            citation_labels=["E01", "E02", "E03"],
        )

        mocked = json.dumps({
            "decomposed": [
                {
                    "parent_id": "C03",
                    "sub_claims": [
                        {"text": "System A is fast.", "citations": ["E01"]},
                        {"text": "System B is accurate.", "citations": ["E02"]},
                        {"text": "System C is cheap.", "citations": ["E03"]},
                    ],
                }
            ]
        })

        with _patch_llm(mocked):
            result = decompose_claims([claim])

        assert len(result) == 3
        assert [r.id for r in result] == ["C03_a", "C03_b", "C03_c"]
        assert all(r.parent_claim_id == "C03" for r in result)

    def test_sub_claim_inherits_all_parent_citations_when_llm_does_not_refine(self):
        """
        The LLM may not be able to narrow citations; in that case both
        sub-claims carry the full parent citation set — this is correct and
        safe (more conservative, not less).
        """
        claim = Claim(
            id="C01",
            text="System A does X and Y.",
            citation_labels=["E01", "E02"],
        )

        mocked = json.dumps({
            "decomposed": [
                {
                    "parent_id": "C01",
                    "sub_claims": [
                        {"text": "System A does X.", "citations": ["E01", "E02"]},
                        {"text": "System A does Y.", "citations": ["E01", "E02"]},
                    ],
                }
            ]
        })

        with _patch_llm(mocked):
            result = decompose_claims([claim])

        assert result[0].citation_labels == ["E01", "E02"]
        assert result[1].citation_labels == ["E01", "E02"]


# ─────────────────────────────────────────────────────────────────────────────
# Test: mixed batches (the common real-world case)
# ─────────────────────────────────────────────────────────────────────────────

class TestDecomposeClaimsMixedBatch:

    def test_mixed_atomic_and_compound_batch(self):
        """
        Batch containing one atomic and one compound claim:
        - Atomic claim C01: returned unchanged, parent_claim_id=None
        - Compound claim C02: split into C02_a / C02_b with parent_claim_id="C02"
        Output is a flat list in original claim order.
        """
        claims = [
            Claim(id="C01", text="ZJIT is written in C.", citation_labels=["E01"]),
            Claim(
                id="C02",
                text="ZJIT compiles whole methods, while YJIT compiles basic blocks.",
                citation_labels=["E02", "E03"],
            ),
        ]

        mocked = json.dumps({
            "decomposed": [
                {
                    "parent_id": "C01",
                    "sub_claims": [{"text": "ZJIT is written in C.", "citations": ["E01"]}],
                },
                {
                    "parent_id": "C02",
                    "sub_claims": [
                        {"text": "ZJIT compiles whole methods.", "citations": ["E02"]},
                        {"text": "YJIT compiles basic blocks.", "citations": ["E03"]},
                    ],
                },
            ]
        })

        with _patch_llm(mocked):
            result = decompose_claims(claims)

        assert len(result) == 3

        c01 = result[0]
        assert c01.id == "C01"
        assert c01.parent_claim_id is None  # atomic, no grouping

        c02a = result[1]
        assert c02a.id == "C02_a"
        assert c02a.parent_claim_id == "C02"
        assert c02a.citation_labels == ["E02"]

        c02b = result[2]
        assert c02b.id == "C02_b"
        assert c02b.parent_claim_id == "C02"
        assert c02b.citation_labels == ["E03"]

    def test_output_order_matches_input_claim_order(self):
        """
        Claims must come out in the same order they went in
        (atomic claims first if they were first, etc.).
        This matters for the frontend, which renders in list order.
        """
        claims = [
            Claim(id="C01", text="Fact one.", citation_labels=["E01"]),
            Claim(id="C02", text="Fact two and three.", citation_labels=["E02"]),
            Claim(id="C03", text="Fact four.", citation_labels=["E03"]),
        ]

        # LLM returns them in order, C02 is split
        mocked = json.dumps({
            "decomposed": [
                {"parent_id": "C01", "sub_claims": [{"text": "Fact one.", "citations": ["E01"]}]},
                {
                    "parent_id": "C02",
                    "sub_claims": [
                        {"text": "Fact two.", "citations": ["E02"]},
                        {"text": "Fact three.", "citations": ["E02"]},
                    ],
                },
                {"parent_id": "C03", "sub_claims": [{"text": "Fact four.", "citations": ["E03"]}]},
            ]
        })

        with _patch_llm(mocked):
            result = decompose_claims(claims)

        assert [r.id for r in result] == ["C01", "C02_a", "C02_b", "C03"]


# ─────────────────────────────────────────────────────────────────────────────
# Test: robustness / error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestDecomposeClaimsRobustness:

    def test_markdown_fences_stripped_before_json_parse(self):
        """LLM sometimes wraps JSON in ```json fences despite instructions."""
        claim = Claim(id="C01", text="ZJIT is fast.", citation_labels=["E01"])

        mocked = "```json\n" + json.dumps({
            "decomposed": [
                {"parent_id": "C01", "sub_claims": [{"text": "ZJIT is fast.", "citations": ["E01"]}]}
            ]
        }) + "\n```"

        with _patch_llm(mocked):
            result = decompose_claims([claim])

        assert len(result) == 1
        assert result[0].id == "C01"

    def test_invalid_json_raises_value_error_with_message(self):
        """Malformed LLM response raises ValueError — not a raw crash — with clear message."""
        claim = Claim(id="C01", text="ZJIT is fast.", citation_labels=["E01"])

        with _patch_llm("This is not JSON at all."):
            with pytest.raises(ValueError, match="Decomposition LLM did not return valid JSON"):
                decompose_claims([claim])

    def test_unknown_parent_id_in_response_is_skipped_gracefully(self):
        """
        If the LLM hallucinates a parent_id that doesn't exist in the input,
        that entry is skipped and remaining valid claims are still processed.
        The original claims without a matching response entry fall back to
        being returned unchanged (defensive: better to not decompose than crash).
        """
        claims = [
            Claim(id="C01", text="Fact one.", citation_labels=["E01"]),
        ]

        # LLM returns C99 which doesn't exist in input
        mocked = json.dumps({
            "decomposed": [
                {"parent_id": "C99", "sub_claims": [{"text": "Hallucinated.", "citations": ["E01"]}]},
                {"parent_id": "C01", "sub_claims": [{"text": "Fact one.", "citations": ["E01"]}]},
            ]
        })

        with _patch_llm(mocked):
            result = decompose_claims(claims)

        # C99 is silently ignored; C01 is returned correctly
        assert len(result) == 1
        assert result[0].id == "C01"
        assert result[0].parent_claim_id is None

    def test_claim_with_no_matching_llm_response_returned_unchanged(self):
        """
        If the LLM omits a claim entirely from its response (e.g. due to
        token truncation), that claim falls back to being returned unchanged
        rather than silently dropped — robustness over correctness here.
        """
        claims = [
            Claim(id="C01", text="Fact one.", citation_labels=["E01"]),
            Claim(id="C02", text="Fact two.", citation_labels=["E02"]),  # LLM omits this
        ]

        # LLM only returns C01
        mocked = json.dumps({
            "decomposed": [
                {"parent_id": "C01", "sub_claims": [{"text": "Fact one.", "citations": ["E01"]}]},
            ]
        })

        with _patch_llm(mocked):
            result = decompose_claims(claims)

        # C02 falls back to original, unchanged, parent_claim_id=None
        assert len(result) == 2
        assert result[0].id == "C01"
        assert result[1].id == "C02"
        assert result[1].text == "Fact two."
        assert result[1].parent_claim_id is None
