import sys
import platform
import importlib


# ============================================================
# REQUIRED PACKAGES
# ============================================================

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "transformers",
    "emoji",
    "tqdm",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_packages():
    """Import each required package and report its version."""

    all_ok = True

    for name in REQUIRED_PACKAGES:

        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"  {name:<14}: {version}")

        except ImportError as error:
            print(f"  {name:<14}: MISSING ({error})")
            all_ok = False

    return all_ok


def select_device():
    """Return 'mps' on Apple Silicon when available, otherwise 'cpu'."""

    import torch

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def check_device(device):
    """Run a tiny tensor operation on the selected device."""

    import torch

    x = torch.ones(2, 3, device=device)
    y = (x * 2).sum().item()

    return y == 12.0


def check_bertweet_tokenizer_support():
    """
    BERTweet's tokenizer normalises tweets using emoji.demojize.
    This checks that both pieces are importable (no model download).
    """

    from transformers import BertweetTokenizer
    from emoji import demojize

    sample = demojize("fake news 😂")

    return BertweetTokenizer is not None and ":" in sample


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NLP ENVIRONMENT CHECK")
    print("=" * 60)

    print(f"\nPython   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print(f"Machine  : {platform.machine()}")

    print("\nPackages:")
    packages_ok = check_packages()

    if not packages_ok:
        print("\nERROR: Missing packages. Run:")
        print("  .venv/bin/pip install -r requirements-nlp.txt")
        sys.exit(1)

    device = select_device()
    device_ok = check_device(device)

    print(f"\nCompute device : {device}")
    print(f"Device test    : {'OK' if device_ok else 'FAILED'}")

    tokenizer_ok = check_bertweet_tokenizer_support()

    print(
        f"BERTweet tokenizer support : "
        f"{'OK' if tokenizer_ok else 'FAILED'}"
    )

    print("\n" + "=" * 60)

    if device_ok and tokenizer_ok:
        print("ENVIRONMENT CHECK PASSED")
    else:
        print("ENVIRONMENT CHECK FAILED")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
