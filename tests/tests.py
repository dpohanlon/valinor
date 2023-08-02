import subprocess as sp

def testWithCLI():

    # cwd: valinor/tests

    args = '--combinationsFile dfCombs_ace.pq '
    args += '--singletonsFile dfSgl_ace.pq '
    args += '--controlsFile dfCalib_ace.pq '
    args += '--epochs 100 '
    args += '-n cliTest '

    sp.check_call(f'python ../valinor/run_model.py {args}', shell = True)

if __name__ == '__main__':

    testWithCLI()
