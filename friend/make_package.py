"""Build the zips others get: the code to build the image, plus scripts; never keys, configs or data.
    python friend/make_package.py                    concilium-friend.zip: the friend's member run (friend/README.md)
    python friend/make_package.py windows            concilium-friend-windows.zip (no .sh files)
    python friend/make_package.py release            concilium-release.zip: every role (release/QUICKSTART.md)
"""
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = ["Dockerfile", ".dockerignore", "requirements.txt", "requirements-research.txt", "deploy.py", "threshold.py", "fl_core.py", "gap_c.py", "gap_b.py",
        "gap_b_cnn.py", "range_proof.py"]
TREES = ["tasks", "pine_poc"]
SKIP = ("__pycache__", "wheels", ".key", ".pub", ".state", ".pem", ".json", ".npz")


WINDOWS = sys.argv[1:] == ["windows"]
RELEASE = sys.argv[1:] == ["release"]
SCRIPTS = "release" if RELEASE else "friend"


def files():
    yield from ((os.path.join(ROOT, f), f) for f in CODE)
    for tree in TREES:
        for d, dirs, names in os.walk(os.path.join(ROOT, tree)):
            dirs[:] = [x for x in dirs if x not in SKIP]
            for n in names:
                if not n.endswith(SKIP):
                    full = os.path.join(d, n)
                    yield full, os.path.relpath(full, ROOT)
    if RELEASE:
        yield from ((os.path.join(ROOT, f), f) for f in ("README.md", "RESEARCH.md", "LICENSE"))
    for n in os.listdir(os.path.join(ROOT, SCRIPTS)):
        if n != "make_package.py" and not (WINDOWS and n.endswith(".sh")):
            yield os.path.join(ROOT, SCRIPTS, n), n  # the scripts sit next to the Dockerfile


if __name__ == "__main__":
    out = os.path.join(ROOT, "concilium-release.zip" if RELEASE else "concilium-friend-windows.zip" if WINDOWS
                       else "concilium-friend.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for full, name in files():
            z.write(full, name)
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert not any(n.endswith((".key", ".pub", ".state", ".json")) for n in names), "no keys or configs"
    assert not (WINDOWS and any(n.endswith(".sh") for n in names)), "windows zip has .sh files"
    print(f"{out}: {len(names)} files, {os.path.getsize(out) / 1e6:.1f} MB")
