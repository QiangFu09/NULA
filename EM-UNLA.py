"""Compatibility entry point; see train.py --help."""

if __name__ == "__main__":
    from train import main

    main(default_methods=["em"])
