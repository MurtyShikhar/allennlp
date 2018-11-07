
#! /usr/bin/env python

# pylint: disable=invalid-name,wrong-import-position,protected-access
import sys
import os
import gzip
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers import LatentAlignmentDatasetReader
from allennlp.models.archival import load_archive

def make_data(input_examples_file: str,
              archived_model_file: str,
              params_file: str) -> None:
    params = Params.from_file(params_file)
    reader = DatasetReader.from_params(params.pop('dataset_reader'))

    dataset = reader.read(input_examples_file)
    archive = load_archive(archived_model_file)
    model = archive.model
    model.eval()


    with open(input_examples_file, "r") as data_file:
        examples = json.load(data_file)

    acc = 0.0
    hitsat10 = 0.0
    hitsat5 = 0.0
    hitsat3 = 0.0
    total = 0.0
    x = []
    for example, instance in zip(examples, dataset):
        utterance, lf = example
        gold_lf = lf[0] # this is how the validation data is organized. see preprocess_latent_alignment.py for more details
        lf.sort(key=len)
        if gold_lf in lf[:10]: hitsat10 += 1.0
        if gold_lf in lf[:5]: hitsat5 += 1.0
        if gold_lf in lf[:3]: hitsat3 += 1.0
        if lf[0] == gold_lf: acc += 1.0

    

        #if len(lf) <= 10: acc += 1.0
        #x.append(len(lf))
        total += 1.0
        #print(instance)
        #outputs = model.forward_on_instance(instance)
        #utterance, sempre_forms = example
        #if outputs["most_similar"] ==  sempre_forms[0]: acc += 1.0
        #total += 1.0

    print(f"Accuracy = {acc/total} - hits@10 = {hitsat10/total} - hits@5= {hitsat5/total} - hits@3 = {hitsat3/total}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", type=str, help="Input file")
    argparser.add_argument("archived_model", type=str, help="Archived model.tar.gz")
    argparser.add_argument("params", type=str, help="Params for creating dataset")
    args = argparser.parse_args()
    make_data(args.input,  args.archived_model, args.params)
