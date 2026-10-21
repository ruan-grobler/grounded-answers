"""Token prices, to put a dollar figure on every question.

List prices in US dollars per million tokens, from Anthropic's published
pricing as of October 2026. Check https://www.anthropic.com/pricing before
relying on them. Cache writes are priced at 1.25x input (5 minute TTL).
"""

from __future__ import annotations

from dataclasses import dataclass

from grounded_answers.llm import Usage

PRICES_AS_OF = "October 2026"
MILLION = 1_000_000


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float

    @property
    def cache_write(self) -> float:
        return self.input * 1.25


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.0, 20.0, 0.20),
    "claude-opus-5": Price(5.0, 25.0, 0.50),
    "claude-sonnet-5-5": Price(2.0, 10.0, 0.20),
    "claude-sonnet-5": Price(2.0, 10.0, 0.20),
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01),
    "claude-fable-5-1": Price(10.0, 50.0, 0.25),
}


def cost_usd(model: str, usage: Usage) -> float | None:
    """Dollar cost of `usage` on `model`, or None if the model has no known price."""
    price = PRICES.get(model)
    if price is None:
        return None
    return (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * price.cache_write
    ) / MILLION
