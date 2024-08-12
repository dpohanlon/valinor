import os

import subprocess as sp

import argparse

import unittest

import os


class Test01ValinorSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up simulation data test class...")

    def testGenerateData(self):
        if not os.path.isdir("tests/data"):
            os.mkdir("tests/data")

        args = "--out-dir tests/data "
        args += "--nCellLines 2 "
        args += "--nGenes 25 "

        sp.check_call(f"valinor_sim {args}", shell=True)


class Test02Valinor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up Valinor test class...")

    def testWithCLI(self):
        # cwd: valinor

        args = "--combinationsFile tests/data/dfCombs_sim.pq "
        args += "--singletonsFile tests/data/dfSgl_sim.pq "
        args += "--controlsFile tests/data/dfCalib_sim.pq "
        args += "--epochs 1 "
        args += "--nSamples 10 "
        args += "-n cliTest "

        sp.check_call(f"valinor {args}", shell=True)

    def testWithCLIPriors(self):
        # cwd: valinor

        args = "--combinationsFile tests/data/dfCombs_sim.pq "
        args += "--singletonsFile tests/data/dfSgl_sim.pq "
        args += "--controlsFile tests/data/dfCalib_sim.pq "
        args += "--epochs 1 "
        args += "--nSamples 10 "
        args += "-n cliTest "
        args += "--priorsFile tests/priors.yaml "

        sp.check_call(f"valinor {args}", shell=True)

    def testWithZINB(self):
        # cwd: valinor

        args = "--combinationsFile tests/data/dfCombs_sim.pq "
        args += "--singletonsFile tests/data/dfSgl_sim.pq "
        args += "--controlsFile tests/data/dfCalib_sim.pq "
        args += "--epochs 1 "
        args += "--nSamples 10 "
        args += "-n zinbTest "
        args += "--ZINB"

        sp.check_call(f"valinor {args}", shell=True)

    def testWithBatching(self):
        # cwd: valinor

        args = "--combinationsFile tests/data/dfCombs_sim.pq "
        args += "--singletonsFile tests/data/dfSgl_sim.pq "
        args += "--controlsFile tests/data/dfCalib_sim.pq "
        args += "--epochs 1 "
        args += "--nSamples 10 "
        args += "--batch-sample "
        args += "--batch-size 1024"
        args += "-n batchTest "

        sp.check_call(f"valinor {args}", shell=True)


class Test03ValinorReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up Valinor test REPORT class...")

    def testWithCLIReport(self):
        # cwd: valinor

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
    suite = unittest.TestSuite()
    for test_class in [TestValinorSim, TestValinor, TestValinorReport]:
        tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    unittest.TextTestRunner().run(suite)
