# src/utils/helpers.py
import struct
import socket

def float_to_ip(f: float) -> str:
    """
    Convert a corrupted float (from 32-bit int IP) back to a valid IPv4 string.
    """
    try:
        # Truncate to 32-bit unsigned int
        ip_int = int(f) & 0xFFFFFFFF
        # Pack to 4 bytes and unpack as IP
        return socket.inet_ntoa(struct.pack('!I', ip_int))
    except Exception as e:
        print(f"Failed to convert {f} to IP: {e}")
        return "0.0.0.0"

def ip_to_int(ip: str) -> int:
    """
    Convert IP string to integer.
    If input is a float string, convert it back to IP first.
    """
    try:
        # If it's a float-like string, convert it back
        if '.' not in ip or 'e' in ip:
            try:
                f = float(ip)
                ip = float_to_ip(f)
            except:
                return 0
        # Now convert valid IP string to int
        parts = ip.strip().split('.')
        if len(parts) == 4 and all(0 <= int(p) <= 255 for p in parts):
            return (int(parts[0]) << 24) + (int(parts[1]) << 16) + (int(parts[2]) << 8) + int(parts[3])
        else:
            return 0
    except:
        return 0