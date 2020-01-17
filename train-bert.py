from modules.data.fre.prc import fact_ru_eval_preprocess
import argparse, os
from shutil import rmtree
from modules.data import bert_data
from modules.models.bert_models import BERTBiLSTMAttnCRF
from modules.train.train import NerLearner

from config import TEST_FILE, DEV_FILE, IDX2LABELS_FILE, CHECKPOINT_FILE, NUM_EPOCHS, TMP_FOLDER

# CKPTS_FOLDER = 'ckpts'
# TMP_FOLDER = 'tmp'
# TEST_FILE = f'{TMP_FOLDER}/test.csv'
# DEV_FILE = f'{TMP_FOLDER}/dev.csv'
# IDX2LABELS_FILE = f'{TMP_FOLDER}/idx2labels4.txt'

# CHECKPOINT_FILE = f"{CKPTS_FOLDER}/fre-BERTBiLSTMAttnCRF.cpt"

# num_epochs = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--fact_ru_eval_root', type=str, default='/home/dima/factRuEval-2016')

    args = parser.parse_args()
    
    if not os.path.isdir(TMP_FOLDER):
        os.mkdir(TMP_FOLDER)

    # 1. Prepare data

    fact_ru_eval_preprocess(f'{args.fact_ru_eval_root}/devset', f'{args.fact_ru_eval_root}/testset', DEV_FILE, TEST_FILE)

    # 2. Preprocess data

    data = bert_data.LearnData.create(
        train_df_path=TEST_FILE,
        valid_df_path=DEV_FILE,
        idx2labels_path=IDX2LABELS_FILE,
        clear_cache=True
    )

    # 3. Create model
    print('Creating the model...')
    model = BERTBiLSTMAttnCRF.create(len(data.train_ds.idx2label), crf_dropout=0.3)

    # 4. Train the model
    print('Prepare training...')
    learner = NerLearner(model, data, CHECKPOINT_FILE, t_total=NUM_EPOCHS * len(data.train_dl))
    print(f'Got {model.get_n_trainable_params()} trainable params')

    print('Training...')
    learner.fit(epochs=NUM_EPOCHS)

    #rmtree(TMP_FOLDER)