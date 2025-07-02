import os
import hashlib
from typing import Callable

# Simple XOR-based cipher for demonstration purposes
# Not secure - only used to provide placeholder implementations

CipherFn = Callable[[bytes], bytes]


def _xor_cipher_factory(key_size: int = 16, nonce_size: int = 16) -> CipherFn:
    """Return a simple XOR cipher function."""
    key = os.urandom(key_size)
    nonce = os.urandom(nonce_size)
    counter = 0

    def encrypt(data: bytes) -> bytes:
        nonlocal counter
        out = bytearray(len(data))
        idx = 0
        while idx < len(data):
            counter_bytes = counter.to_bytes(8, "big")
            stream = hashlib.sha256(key + nonce + counter_bytes).digest()
            for b in stream:
                if idx >= len(data):
                    break
                out[idx] = data[idx] ^ b
                idx += 1
            counter += 1
        return bytes(out)

    return encrypt


def get_cipher(algo: str) -> CipherFn:
    """Return a cipher function for the given algorithm name."""
    if algo in ("AES-128", "Blowfish", "Twofish", "ASCON-128"):
        return _xor_cipher_factory(16)
    if algo in ("AES-256", "Camellia-256", "ChaCha20"):
        # 32-byte key
        return _xor_cipher_factory(32)
    # Fallback
    return _xor_cipher_factory(16)
