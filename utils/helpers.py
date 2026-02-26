import argparse

def parse_arguments(description, args):
    parser = argparse.ArgumentParser(
        description=description
    )

    for arg, help in args.items():
        parser.add_argument(arg, help=help)

    return parser.parse_args()

def sod_to_hms(sod):
    sod = float(sod)

    h = int(sod // 3600)
    m = int((sod % 3600) // 60)
    s = sod % 60  # keep fractional part

    return f"{h:02d}:{m:02d}:{s:06.3f}"  # 2 digits + .mmm
