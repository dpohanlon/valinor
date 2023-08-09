import subprocess as sp


def testWithCLI():
    # cwd: valinor/tests

    args = "--combinationsFile tests/data/dfCombs_ace.pq "
    args += "--singletonsFile tests/data/dfSgl_ace.pq "
    args += "--controlsFile tests/data/dfCalib_ace.pq "
    args += "--epochs 100 "
    args += "-n cliTest "

    sp.check_call(f"valinor {args}", shell=True)


if __name__ == "__main__":
    testWithCLI()
