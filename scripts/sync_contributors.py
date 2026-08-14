#!/usr/bin/env python3
"""
SignBridge — Automatic Git Contributor Sync Script
Reads `git shortlog` and updates `SIGNATURE.json` contributor list while maintaining rjb07 as Lead.
"""

import json
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SIG_FILE = ROOT_DIR / "SIGNATURE.json"

def sync():
    if not SIG_FILE.exists():
        print(f"[!] Warning: {SIG_FILE} not found.")
        return

    sig_data = json.loads(SIG_FILE.read_text(encoding="utf-8"))
    
    # Lead must always be rjb07 from Ultima Core AI Labs
    sig_data["lead"] = "rjb07"
    sig_data["organization"] = "Ultima Core AI Labs"

    try:
        out = subprocess.check_output(["git", "shortlog", "-sn", "HEAD"], cwd=ROOT_DIR, text=True)
        lines = out.strip().split("\n")
        
        contribs = [
            {
                "name": "rjb07",
                "role": "Lead Architect & Creator",
                "organization": "Ultima Core AI Labs",
                "date": "2026-08-14"
            }
        ]

        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                name = parts[1]
                if name.lower() != "rjb07" and not any(c["name"] == name for c in contribs):
                    contribs.append({
                        "name": name,
                        "role": "Open-Source Contributor",
                        "organization": "Community",
                        "date": "2026-08-14"
                    })

        sig_data["contributors"] = contribs
        with open(SIG_FILE, "w", encoding="utf-8") as f:
            json.dump(sig_data, f, indent=2)

        print(f"[+] Synced {len(contribs)} contributor(s) into SIGNATURE.json")
    except Exception as e:
        print(f"[!] Warning: Git log sync skipped ({e}). Preserving SIGNATURE.json.")

if __name__ == "__main__":
    sync()
