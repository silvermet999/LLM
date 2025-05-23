from query import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def test_rag():
    assert query_val(question="What is article 7 about? (Answer with one word)",
    expected = "Irregularities")

def query_val(question: str, expected: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)

    print(prompt)

    if "true" in evaluation_results_str:
        print("\033[92m" + f"Response: {evaluation_results_str}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str:
        print("\033[91m" + f"Response: {evaluation_results_str}" + "\033[0m")
        return False
    else:
        raise ValueError(
        )
