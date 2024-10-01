from enum import Enum


class Activation(Enum):
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"

    @staticmethod
    def from_str(s: str) -> "Activation":
        if s == "relu":
            return Activation.RELU
        elif s == "gelu":
            return Activation.GELU
        elif s == "silu":
            return Activation.SILU
        else:
            raise ValueError(f"Unknown activation function: {s}")
