# Blockers & Environment Limitations

## BLK-01: Ragas LLM Evaluation API Key Requirement
- **Description**: Automated evaluation with Ragas (`faithfulness`, `answer_relevancy`, `answer_correctness`) requires a live LLM endpoint for its judge evaluator. In offline test environments without a funded Google Gemini API key (`GOOGLE_API_KEY`), live LLM evaluations against Google's API returned `400 INVALID_ARGUMENT (API_KEY_INVALID)`.
- **Handling & Fallback**: As specified in SPEC-01 §12.3 ("If the adapter fails... record ragas: 'skipped' with the error in reports/blockers.md, and still report gold_answer_contained"), the Ragas adapter logged the condition and evaluated deterministic `gold_answer_contained` (achieving 66.50% exact span containment and 98.00% top-5 gold document recall).

