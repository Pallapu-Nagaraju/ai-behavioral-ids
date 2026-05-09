"""
core/normalizer.py — Command Normalization
Strips IPs, paths, URLs → canonical form for consistent feature extraction.
"""

import re


def normalize_command(cmd: str) -> str:
    """
    Normalize a raw command line for consistent feature extraction.
    Order matters: do URL substitution before path substitution.
    """
    cmd = cmd.strip().lower()

    # Replace IP addresses
    cmd = re.sub(r'\b\d{1,3}(\.\d{1,3}){3}\b', '<IP>', cmd)

    # Replace user@<IP> patterns
    cmd = re.sub(r'\b\w+@<IP>', '<USER>@<IP>', cmd)

    # Handle URLs BEFORE generic path replacement
    if "http://" in cmd or "https://" in cmd:
        cmd = re.sub(r'https?://<IP>/\S*', 'http://<IP>/<PATH>', cmd)
        cmd = re.sub(r'https?://\S+',      'http://<HOST>/<PATH>', cmd)
        return cmd

    # Replace absolute paths (Unix)
    cmd = re.sub(r'/[^\s]+', '<PATH>', cmd)

    # Replace Windows paths
    cmd = re.sub(r'[a-zA-Z]:\\\S+', '<PATH>', cmd)

    return cmd
