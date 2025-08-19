from src.utils.helpers import ip_to_int

def test_ip_to_int():
    assert ip_to_int("192.168.1.1") == 3232235777
    assert ip_to_int("invalid") == 0