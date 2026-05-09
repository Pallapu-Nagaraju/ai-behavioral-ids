from core.normalizer import normalize_command
test_commands = [
    "ssh bob@192.168.1.101",
    "scp file.txt user@10.0.0.5:/tmp",
    "curl http://192.168.1.50/data",
    "cd /home/user/project",
    "C:\\Users\\test\\file.txt"
]

for cmd in test_commands:
    print("Original :", cmd)
    print("Normalized:", normalize_command(cmd))
    print("-" * 40)