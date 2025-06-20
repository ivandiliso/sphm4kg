from timeit import default_timer as timer


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def pretty_print(str, c):
    print(f"{c}{str}{color.END}")


class SimpleLogger:
    def __init__(self):
        self.time = 0
        self.str = ""

    def start(self, str=""):
        self.str = str
        print(f"[{color.RED}START{color.END}] {str} ...", end="\r")
        self.time = timer()

    def end(self):
        self.time = timer() - self.time
        print(
            f"[{color.GREEN}DONE{color.END} ] {self.str} in {color.CYAN}{self.time:09.4f}{color.END}s",
            end="\n",
        )
