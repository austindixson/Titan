from titan.config import RetryConfig
from titan.provider import retry_call, ProviderError


def test_retry_then_success():
    state = {"n": 0}

    def fn():
        state["n"] += 1
        if state["n"] < 3:
            raise ProviderError("temp", retryable=True)
        return 42

    out = retry_call(fn, RetryConfig(max_retries=3, base_delay_ms=1, max_delay_ms=2))
    assert out == 42


def test_non_retryable_fails_fast():
    def fn():
        raise ProviderError("bad", retryable=False)

    try:
        retry_call(fn, RetryConfig(max_retries=3, base_delay_ms=1, max_delay_ms=2))
        assert False, "must fail"
    except ProviderError as e:
        assert str(e) == "bad"
