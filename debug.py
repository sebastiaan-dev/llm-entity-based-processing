from colorama import Fore, Style


def print_info(message: str, value: str = None):
    print(
        f"{Fore.BLUE}{Style.BRIGHT}[INFO] {Fore.WHITE}{message} {Style.NORMAL}{value}"
    )


def print_debug(message: str, value: str = None):
    print(
        f"{Fore.CYAN}{Style.BRIGHT}[DEBUG] {Fore.WHITE}{message} {Style.NORMAL}{value}"
    )


def print_warn(message: str, value: str = None):
    print(
        f"{Fore.YELLOW}{Style.BRIGHT}[WARNING] {Fore.WHITE}{message} {Style.NORMAL}{value}"
    )


def print_error(message: str, value: str = None):
    print(
        f"{Fore.RED}{Style.BRIGHT}[ERROR] {Fore.WHITE}{message} {Style.NORMAL}{value}"
    )
