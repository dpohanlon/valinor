import os

import subprocess as sp

import argparse

import unittest

import os


# def testWithCLI():
#     # cwd: valinor

#     args = "--combinationsFile tests/data/dfCombs_sim.pq "
#     args += "--singletonsFile tests/data/dfSgl_sim.pq "
#     args += "--controlsFile tests/data/dfCalib_sim.pq "
#     args += "--epochs 10 "
#     args += "--nSamples 10 "
#     args += "-n cliTest "

#     sp.check_call(f"valinor {args}", shell=True)

#     args = "--combinationsFile tests/data/dfCombs_sim.pq "
#     args += "--singletonsFile tests/data/dfSgl_sim.pq "
#     args += "--controlsFile tests/data/dfCalib_sim.pq "
#     args += "--epochs 10 "
#     args += "--nSamples 10 "
#     args += "-n cliTest "
#     args += "--priorsFile tests/priors.yaml "

#     sp.check_call(f"valinor {args}", shell=True)

# def testWithCLIReport():
#     # cwd: valinor


# def testGenerateData():
#     # Call the simulator to generate some data for our tests

#     if not os.path.isdir("tests/data"):
#         os.mkdir("tests/data")

#     args = "--out-dir tests/data "


#     sp.check_call(f"valinor_sim {args}", shell=True)


class Test01ValinorSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up simulation data test class...")

    def testGenerateData(self):
        # Call the simulator to generate some data for our tests

        if not os.path.isdir("tests/data"):
            os.mkdir("tests/data")

        args = "--out-dir tests/data "

        sp.check_call(f"valinor_sim {args}", shell=True)


class Test02Valinor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This method runs once before the first test of this class
        print("Setting up Valinor test class...")

    def testWithCLI(self):
        # cwd: valinor

        args = "--combinationsFile tests/data/dfCombs_sim.pq "
        args += "--singletonsFile tests/data/dfSgl_sim.pq "
        args += "--controlsFile tests/data/dfCalib_sim.pq "
        args += "--epochs 10 "
        args += "--nSamples 10 "
        args += "-n cliTest "

        sp.check_call(f"valinor {args}", shell=True)

    def testWithCLIPriors(self):
        # cwd: valinor

        args = "--combinationsFile tests/data/dfCombs_sim.pq "
        args += "--singletonsFile tests/data/dfSgl_sim.pq "
        args += "--controlsFile tests/data/dfCalib_sim.pq "
        args += "--epochs 10 "
        args += "--nSamples 10 "
        args += "-n cliTest "
        args += "--priorsFile tests/priors.yaml "

        sp.check_call(f"valinor {args}", shell=True)


class Test03ValinorReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This method runs once before the first test of this class
        print("Setting up Valinor test REPORT class...")

    def testWithCLIReport(self):
        # cwd: valinor

        directory_path = "."  # Replace with your directory path
        list_directory_contents(directory_path)

        directory_path = "tests/"  # Replace with your directory path
        list_directory_contents(directory_path)

        directory_path = "tests/data/"  # Replace with your directory path
        list_directory_contents(directory_path)

        args = "--val_combo combsModel_cliTest.pq "
        args += "--val_single singlesModel_cliTest.pq "
        args += "--data_combo tests/data/dfCombs_sim.pq "
        args += "--data_single tests/data/dfSgl_sim.pq "
        args += "--output_file tests/valinoroutput_processed_cliTest.pq "
        args += "--report_folder tests/valinorreport_cliTest "
        args += "--valinorLossFile valinor_loss_cliTest.svg "
        args += "--valinorConfigFile valinorrun_cliTest.json "
        args += "--subsetSLpairs"

        sp.check_call(f"valinorreport {args}", shell=True)


if __name__ == "__main__":
    # unittest test runner will automatically find all classes that inherit from unittest.TestCase in the module
    suite = unittest.TestSuite()
    for test_class in [Test01ValinorSim, Test02Valinor, Test03ValinorReport]:
        tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    unittest.TextTestRunner().run(suite)
    # argParser = argparse.ArgumentParser()

    # argParser.add_argument(
    #     "--generate-input",
    #     action="store_true",
    #     dest="generate_input",
    #     default=False,
    #     help="Generate a simulated input dataset for integration testing.",
    # )

    # args = argParser.parse_args()

    # if args.generate_input:
    #     testGenerateData()

    # else:
    #     testWithCLI()
    #     testWithCLIPriors()
