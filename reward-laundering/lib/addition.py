"""2-digit addition: generation and exact-match verification (the easy main task)."""

from __future__ import annotations

import random

import attrs


@attrs.frozen
class AdditionInstance:
    a: int
    b: int

    @property
    def answer(self) -> int:
        return self.a + self.b

    def key(self) -> tuple[int, int]:
        return (self.a, self.b)


def generate_addition(rng: random.Random) -> AdditionInstance:
    """Two operands sampled uniformly from 10..99 (inclusive)."""
    return AdditionInstance(a=rng.randint(10, 99), b=rng.randint(10, 99))


def verify_addition(instance: AdditionInstance, parsed_answer: int | None) -> bool:
    return parsed_answer is not None and parsed_answer == instance.answer
