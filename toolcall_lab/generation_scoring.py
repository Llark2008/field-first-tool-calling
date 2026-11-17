"""Score a finished generation while enforcing the fixed token budget policy."""
from .evaluation import score_response


def score_generation(case, text, finish_reason):
    result = score_response(case, text)
    result['truncated'] = finish_reason == 'length'
    if result['truncated']:
        result['success'] = False
        result['error'] = 'truncated: output token limit reached'
    return result
