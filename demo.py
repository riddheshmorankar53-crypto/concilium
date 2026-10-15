"""Runs the whole story in order. Every script asserts its own claims, so a clean run is the proof.
    python demo.py                   # every step, ~30 min (Docker + `docker build -t concilium-research .`)
    python demo.py gap_d threshold   # just some steps
"""
import os
import subprocess
import sys
import time

# name: (what it shows, script, environment)
STEPS = {
    "fl_core": ("1. Encrypted FedAvg + poisoning filter: same model as plaintext, attack stopped.", "fl_core", {}),
    "gap_b": ("2. But encrypted FLTrust approximates 1/|u|: updates just outside its range break it.", "gap_b", {}),
    "gap_b_cnn": ("3. Same attack on a chest X-ray CNN: pneumonia disappears. A 1-bit range gate fixes it.",
                  "gap_b_cnn", {}),
    "gap_c": ("4. The key holder is a decryption oracle: the server reads one client's update. Guard it.",
              "gap_c", {}),
    "gap_d": ("5. Worse: one decrypted aggregate leaks the secret key. Only decryption-time noise stops it.",
              "gap_d", {}),
    "gap_e": ("6. And an attacker who knows the filter? Clipping to the median length holds up, non-IID too.",
              "gap_e", {}),
    "gap_g": ("6b. Attackers who optimize against the filter (literature attacks + one tailored to its angle check): "
              "it holds; a leaked reference doesn't.", "gap_g", {}),
    "threshold": ("7. Fix it all: 3 committee members share the key, range check on ciphertexts, nobody decrypts "
                  "alone.", "threshold", {}),
    "gap_f": ("8. One oversized update breaks the packed score decryption for everyone. Fall back per participant.",
              "gap_f", {}),
    "flood_probe": ("9. Gap D in our own design: the decryption noise must outweigh the ciphertext's error. Measure "
                    "the error, size the noise by OpenFHE's rule.", "flood_probe", {}),
    "range_proof": ("10. That only holds for bounded updates: participants prove |u| <= B (PINE), and a ciphertext "
                    "that differs from its proof is caught.", "range_proof", {}),
    "deploy": ("11. The same protocol as 14 separate containers over HTTP: real traffic and timing; range proofs reject "
               "a participant lying about its norm; a member crashes.",
               "deploy", {}),
    "failover": ("12. The lead dies mid-round: its successor rules it gone, the round is abandoned, the run goes on.",
                 "deploy", {"CRASH": "lead-mid"}),
    "rejoin": ("13. A member crashes, restarts 30 s later and rejoins from its saved key share.", "deploy",
               {"CRASH": "member-rejoin"}),
    "bad_posts": ("14. A participant posts junk, then a copy of another's post: members drop them, the run goes on "
                  "(gap H).", "deploy", {"BAD_POST": "1"}),
    "resume": ("15. The lead and its successor crash together: too few members left to decrypt, the run stops for "
               "everyone; it goes on as a new run from its last signed model.", "deploy", {"RESUME": "1"}),
}
DOCKER = {"threshold", "gap_f", "flood_probe", "range_proof"}  # OpenFHE's wheel is Linux-only


def command(script):
    if script in DOCKER:
        return ["docker", "run", "--rm", "-v", f"{os.getcwd()}:/work", "-w", "/work", "concilium-research", "python3", f"{script}.py"]
    return [sys.executable, f"{script}.py"]


if __name__ == "__main__":
    names = sys.argv[1:] or list(STEPS)
    unknown = set(names) - set(STEPS)
    if unknown:
        sys.exit(f"unknown steps {sorted(unknown)}; choose from {list(STEPS)}")
    for name in names:
        what, script, env = STEPS[name]
        print(f"\n{'=' * 100}\n{what}\n{'=' * 100}", flush=True)
        t = time.perf_counter()
        if subprocess.run(command(script), env=os.environ | env).returncode:
            sys.exit(f"\n{name} FAILED")
        print(f"[{name}: passed in {time.perf_counter() - t:.0f}s]", flush=True)
    print(f"\nAll {len(names)} steps passed.")
