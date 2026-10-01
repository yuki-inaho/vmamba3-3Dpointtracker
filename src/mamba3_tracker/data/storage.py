"""Disk accounting while background workers add and evict cache files."""

import stat


def tree_bytes(root):
    total = 0
    for path in root.rglob("*"):
        try:
            info = path.stat()
        except FileNotFoundError:
            continue  # A concurrent cache LRU evicted this file.
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total
