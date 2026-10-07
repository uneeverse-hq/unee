"""The `hf` command over IPv4 only, for networks where IPv6 connections to huggingface.co get reset.

    .venv/Scripts/python tools/hf_ipv4.py auth login
    .venv/Scripts/python tools/hf_ipv4.py upload uneeverse/unee-0.8b models/hf/unee-0.8b .

Measured on this laptop's network, 2026-10-07: 5 of 8 default connections from Python succeeded ("WinError 10054,
connection forcibly closed" during the TLS handshake); 8 of 8 over IPv4. That is what broke `hf auth login`, which
checks the token online before saving it. Every argument is passed to `hf` unchanged.
"""

import socket

_getaddrinfo = socket.getaddrinfo


def _ipv4_first(host, *args, **kwargs):
    found = _getaddrinfo(host, *args, **kwargs)
    return [a for a in found if a[0] == socket.AF_INET] or found


socket.getaddrinfo = _ipv4_first

if __name__ == "__main__":
    from huggingface_hub.cli.hf import main

    main()
