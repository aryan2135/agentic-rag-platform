# tests/test_cross_encoder_network.py
import pytest
from app.deps import CrossEncoderScorer


@pytest.mark.network
async def test_cross_encoder_network():
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    scorer = CrossEncoderScorer(model_name)

    query = "How many people live in Paris France?"
    relevant_text = "The population of Paris was estimated at over 2 million inhabitants."
    unrelated_text = "Quantum computing relies on qubits to perform superposition calculations."

    scores = await scorer.ascore(query, [relevant_text, unrelated_text])
    assert len(scores) == 2
    for s in scores:
        assert 0.0 <= s <= 1.0

    # Relevant pair scores > 0.5, unrelated pair scores < 0.5
    assert scores[0] > 0.5
    assert scores[1] < 0.5

