import pytest

from ckks_param_tool.params import SECURITY_TABLES, security_level

# values from Microsoft SEAL native/src/seal/util/hestdparms.h (HE Standard, ternary secret,
# classical attacker)
SEAL_TABLE = {
    128: {4096: 109, 8192: 218, 16384: 438, 32768: 881},
    192: {4096: 75, 8192: 152, 16384: 305, 32768: 611},
    256: {4096: 58, 8192: 118, 16384: 237, 32768: 476},
}


def test_tables_match_seal():
    assert SECURITY_TABLES == SEAL_TABLE


@pytest.mark.parametrize("n,bits,expected", [
    (8192, 218, 128),
    (8192, 219, 0),
    (8192, 152, 192),
    (8192, 153, 128),
    (8192, 118, 256),
    (16384, 305, 192),
    (16384, 237, 256),
    (32768, 611, 192),
    (32768, 612, 128),
    (32768, 476, 256),
])
def test_security_level_boundaries(n, bits, expected):
    assert security_level(n, bits) == expected
