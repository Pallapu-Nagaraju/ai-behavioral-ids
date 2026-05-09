import os

def read_bash_history(path=None):
    path = path or os.path.expanduser("~/.bash_history")
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

def stream_stdin():
    while True:
        try:
            line = input()
            if line.strip():
                yield line.strip()
        except EOFError:
            break
