import os

import subprocess as sp

import argparse


def testWithCLI():
    # cwd: valinor

    args = "--combinationsFile tests/data/dfCombs_sim.pq "
    args += "--singletonsFile tests/data/dfSgl_sim.pq "
    args += "--controlsFile tests/data/dfCalib_sim.pq "
    args += "--epochs 10 "
    args += "--nSamples 10 "
    args += "-n cliTest "

    sp.check_call(f"valinor {args}", shell=True)

def testWithCLIPriors():
    # cwd: valinor

    args = "--combinationsFile tests/data/dfCombs_sim.pq "
    args += "--singletonsFile tests/data/dfSgl_sim.pq "
    args += "--controlsFile tests/data/dfCalib_sim.pq "
    args += "--epochs 10 "
    args += "--nSamples 10 "
    args += "-n cliTest "
    args += "--priorsFile tests/priors.yaml "

    sp.check_call(f"valinor {args}", shell=True)

def testGenerateData():
    # Call the simulator to generate some data for our tests

    if not os.path.isdir("tests/data"):
        os.mkdir("tests/data")

    args = "--out-dir tests/data "

    sp.check_call(f"valinor_sim {args}", shell=True)

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "--generate-input",
        action="store_true",
        dest="generate_input",
        default=False,
        help="Generate a simulated input dataset for integration testing.",
    )

    args = argParser.parse_args()

    if args.generate_input:
        testGenerateData()

    else:
        testWithCLI()
        testWithCLIPriors()
