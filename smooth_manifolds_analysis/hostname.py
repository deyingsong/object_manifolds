"""
Hostname utilities module.

Provides utilities for getting system hostname in a persistent manner.
"""

import subprocess
from functools import lru_cache


@lru_cache(maxsize=1)
def get_hostname() -> str:
    """
    Get the system hostname in a persistent, consistent way.
    
    Returns
    -------
    str
        The system hostname
    """
    try:
        result = subprocess.run(['hostname'], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return ""
