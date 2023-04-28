To create the python virtual env, run "make".

This will also install any packages listed in the requirements.txt folder.

Running "make clean" will delete the venv folder.

This makefile works for Windows OS.

MLP.py:
    Best test result achieved was ~54% test accuracy.
    Parameters used:
        LEARNING_RATE = 15e-4
        MOMENTUM = 0.9
        BATCH_SIZE = 64

CNN.py:
    Best test result achieved was >65% test accuracy.
    Parameters used:
        LEARNING_RATE = 1e-2
        MOMENTUM = 0.95
        BATCH_SIZE = 128