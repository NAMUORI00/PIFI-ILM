"""
Classification Arguments - Wrapper for core.arguments
This maintains backward compatibility with existing imports
"""

# Import from core.arguments
from core.arguments import ArgParser, parse_bool

# Expose
__all__ = ['ArgParser', 'parse_bool']
