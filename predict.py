from modules.data.fre.prc import fact_ru_eval_preprocess
import argparse, os
from shutil import rmtree
from nltk.tokenize import RegexpTokenizer

from modules.data import bert_data
from modules.models.bert_models import BERTBiLSTMAttnCRF
from modules.train.train import NerLearner
from modules.data.bert_data import get_data_loader_for_predict
from modules.analyze_utils.utils import bert_labels2tokens, voting_choicer
from modules.analyze_utils.plot_metrics import get_bert_span_report

from file_operations import read, write_lines
from config import TEST_FILE, DEV_FILE, IDX2LABELS_FILE, CHECKPOINT_FILE, NUM_EPOCHS, TMP_FOLDER, RAW_INTERNAL_FILE
from converters import raw_text_to_internal_format


def write_prediction_results(labels, tokens, output_file):
    lines = []
    for sentence_tokens, sentence_labels in zip(tokens, labels):
        for token, label in zip(sentence_tokens, sentence_labels):
            lines.append(f'{token} {label.split("_")[1].capitalize()}')
    write_lines(output_file, lines)

if __name__ == "__main__":
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, default='raw.txt')
    parser.add_argument('--output_file', type=str, default='raw.predictions.txt')

    args = parser.parse_args()

    if not os.path.isdir(TMP_FOLDER):
        os.mkdir(TMP_FOLDER)

    raw_text_to_internal_format(args.input_file, RAW_INTERNAL_FILE, tokenizer)


    # 1. Preprocess data
    data = bert_data.LearnData.create(
        train_df_path=DEV_FILE,
        valid_df_path=TEST_FILE,
        idx2labels_path=IDX2LABELS_FILE,
        clear_cache=True
    )

    dl = get_data_loader_for_predict(data, df_path=RAW_INTERNAL_FILE)

    # 2. Create and load the model
    print('Creating the model...')
    model = BERTBiLSTMAttnCRF.create(len(data.train_ds.idx2label), crf_dropout=0.3)
    learner = NerLearner(model, data, CHECKPOINT_FILE, t_total=NUM_EPOCHS * len(data.train_dl))
    learner.load_model(CHECKPOINT_FILE)

    # 3. Predict
    print('Making predictions...')
    preds = learner.predict(dl)
    pred_tokens, pred_labels = bert_labels2tokens(dl, preds)

    write_prediction_results(pred_labels, pred_tokens, args.output_file)

    #rmtree(TMP_FOLDER)