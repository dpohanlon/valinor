import argparse

def prepareData(data_files, only_singletons = False, no_singletons = False, no controls = True):

    pass

def runValinor(lengths, indices, prior_params, data, name = '', no_singletons=False, only_singletons=False, no_controls = False):

    pass

if __name__ == "__main__":

    # I'd like an argument, please
    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "--no-singletons",
        action="store_true",
        dest="no_singletons",
        default=False,
        help="No singletons.",
    )

    argParser.add_argument(
        "--only-singletons",
        action="store_true",
        dest="only_singletons",
        default=False,
        help="Only singletons.",
    )

    argParser.add_argument(
        "--no-controls",
        action="store_true",
        dest="no_controls",
        default=False,
        help="No controls.",
    )

    argParser.add_argument("-n", type=str, dest="name", default="", help="Output name.")

    args = argParser.parse_args()

    lengths, indices, prior_params, data = prepareData(args.only_singletons)

    runValinor(
        lengths, indices, prior_params, data, args.name, args.no_singletons, args.only_singletons, args.no_controls
    )
