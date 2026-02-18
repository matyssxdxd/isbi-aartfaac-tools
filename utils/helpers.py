import argparse

def parse_arguments(description, args):
    parser = argparse.ArgumentParser(
        description=description
    )

    for arg, help in args.items():
        parser.add_argument(arg, help=help)

    return parser.parse_args()

