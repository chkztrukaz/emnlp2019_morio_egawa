This project provides the code of BiLSTM-CRF (BLC) model which is described in our paper.


Requirements:

- Python 3.5.2+
- As shown in "requirements.txt", our project depends on following libs:
    chainer
    conllu
    h5py
    numpy
    Pillow
    scikit-learn


Run TRAIN:

To run the train program, you just conduct "blc_trainer.py" as follows:
    "python blc_trainer.py"

Variable options such as GPU configuration are available, i.e.,
    "python blc_trainer.py --gpu 2"  # use GPU No.2

In the default option, log files will be generated in /log/blc_log/.
For more options, please refer the argpase in the code.


Run TEST:

To run the test program, you just conduct "blc_tester.py" as follows:
    "python blc_tester.py"

The program above will load our trained BLC model which is located in /trained_model/best_model
Default example sentences will be parsed by the tester.
If you want to change the sentences, please refer the code where you will see a sentence list to change.
