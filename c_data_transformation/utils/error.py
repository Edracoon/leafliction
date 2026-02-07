import sys


RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'


def error(message: str):
    """
    Print an error message and exit the program.
    """
    print(f"{RED}{BOLD}Error: {message}{RESET}")
    sys.exit(1)
