import jsonlines
import re
from tqdm import tqdm
import os
from score_parser_spans import load_spans
import argparse


def get_config():
    """
    Configuration for which token processing will be computed.
    """
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("--dataset", type=str, default='dev')
    config_parser.add_argument("--ner", type=str, default="", help="all, or list of types")
    config_parser.add_argument("--parser_preds", type=int, default=-1, help="attach on parser preds with top-k categories")
    config_parser.add_argument("--use_na_spans", action="store_true", help="attach on mentions not captured by parser")
    return config_parser.parse_args()


def convert_bert_word(word):
    """
    Convert bert token to regular word.
    """
    return re.sub("##|\[SEP\]|\[CLS\]", "", word)


def adjust_cluster_indices(clusters, subtoken_map, sent_start, sent_end):
    """
    Adjust cluster indices to reflect their position within an individual sentence.
    """
    adjusted_clusters = []
    for cluster in clusters:
        for span in cluster:
            if span[0] >= sent_start and span[1] <= sent_end:
                adjusted_start = subtoken_map[span[0]] - subtoken_map[sent_start]
                adjusted_end = subtoken_map[span[1]] - subtoken_map[sent_start]
                adjusted_clusters.append((adjusted_start, adjusted_end))
    return adjusted_clusters


def adjust_top_mentions(mentions, subtoken_map, sent_start, sent_end):
    """
    Adjusts the top mentions indices to reflect position within an individual sentence.
    """
    adjusted_mentions = []
    for mention in mentions:
        if mention[0] >= sent_start and mention[1] <= sent_end:
            adjusted_start = subtoken_map[mention[0]] - subtoken_map[sent_start]
            adjusted_end = subtoken_map[mention[1]] - subtoken_map[sent_start]
            adjusted_mentions.append((adjusted_start, adjusted_end))
    return adjusted_mentions


def num_speakers(speaker_lists):
    """
    Computes number of speakers in a document.
    """
    flat_speaker_list = [speaker for speaker_list in speaker_lists for speaker in speaker_list]
    speaker_set = set(flat_speaker_list)
    speaker_set.discard("[SPL]")
    speaker_set.discard("-")
    return max(1, len(speaker_set))


def add_parser_preds(args, mapped_outputs):
    """
    Adds parser predictions to the output dictionary.
    """
    parser_spans = []
    dev_file = "data/{}_preds/{}.{}.preds".format(args.dataset, args.dataset, args.parser_preds)
    if os.path.exists(dev_file):
        with open(dev_file, 'r') as f:
            for line in f:
                if line.strip():
                    parser_spans.append(load_spans(line))
                else:
                    parser_spans.append(set())
        assert len(parser_spans) == len(mapped_outputs)
        for i, output in enumerate(mapped_outputs):
            output.update({'parser_clusters': list(parser_spans[i])})
    else:
        raise ValueError("Cannot find dev preds at path: {}".format(dev_file))


def add_na_spans(args, mapped_outputs):
    """
    Adds mentions that do not correspond to any node in the tree to the output dictionary.
    """
    na_file = "data/{}.na.spans".format(args.dataset)
    na_spans = []
    if os.path.exists(na_file):
        with open(na_file, 'r') as f:
            for line in f:
                if line.strip():
                    na_spans.append(load_spans(line))
                else:
                    na_spans.append(set())
        assert len(na_spans) == len(mapped_outputs)
        for i, output in enumerate(mapped_outputs):
            output.update({'na_spans': list(na_spans[i])})
    else:
        raise ValueError("Cannot find NA spans at path: {}".format(na_file))


def convert_bert_tokens(args, outputs):
    """
    Converts BERT tokens into a readable format for the parser, i.e. using Penn Treebank tokenization scheme.
    Does the heavy lifting for this script.
    """
    mapped_outputs = []  # Will hold the final results: sentences and mapped span indices
    for output in tqdm(outputs):
        comb_text = [word for sentence in output['sentences'] for word in sentence]
        sentence_start_idx = 0
        sent_so_far = []
        word_so_far = []
        sentence_map = output['sentence_map']
        subtoken_map = output['subtoken_map']
        clusters = output['clusters']
        preds = output['predicted_clusters']
        top_mentions = output['top_spans']
        for i, subword in enumerate(comb_text):
            if i != 0 and sentence_map[i - 1] != sentence_map[i]:  # New sentence
                sent_so_far.append(convert_bert_word(''.join(word_so_far)))
                word_so_far = []
                mapped_outputs.append({'doc_key': output['doc_key'],
                                       'num_speakers': num_speakers(output['speakers']),
                                       'words': sent_so_far,
                                       'clusters': adjust_cluster_indices(clusters, subtoken_map, sentence_start_idx, i - 1),
                                       'predicted_clusters': adjust_cluster_indices(preds, subtoken_map, sentence_start_idx, i - 1),
                                       'top_mentions': adjust_top_mentions(top_mentions, subtoken_map, sentence_start_idx, i - 1)})
                sent_so_far = []
                sentence_start_idx = i
            elif i != 0 and subtoken_map[i - 1] != subtoken_map[i]:  # New word
                fullword = ''.join(word_so_far)
                if fullword != '[SEP][CLS]':  # Need this because sentences indices increment at SEP and CLS tokens
                    sent_so_far.append(convert_bert_word(fullword))
                else:
                    sentence_start_idx += 2  # The sentence actually starts two tokens later due to [SEP] and [CLS]
                word_so_far = []

            word_so_far.append(subword)
    return mapped_outputs


if __name__ == "__main__":
    args = get_config()
    outputs = []
    with jsonlines.open("data/{}.output.english.512.jsonlines".format(args.dataset)) as reader:
        for line in reader:
            outputs.append(line)

    mapped_outputs = convert_bert_tokens(args, outputs)

    with jsonlines.open('data/{}.adjust_span_sents.jsonlines'.format(args.dataset), mode='w') as w:
        for output in mapped_outputs:
            w.write(output)

    with open('data/{}.spans'.format(args.dataset), mode='w') as w:
        for output in mapped_outputs:
            w.write(" ".join([str(idx) for span in output['clusters'] for idx in span]) + '\n')

    with open('data/{}.raw_tokens.sentences'.format(args.dataset), mode='w') as f:
        for output in mapped_outputs:
            f.write(' '.join(output['words']).strip() + '\n')

    # Add on parser predictions
    if args.parser_preds != -1:
        add_parser_preds(args, mapped_outputs)

    if args.use_na_spans:
        add_na_spans(args, mapped_outputs)

    if args.parser_preds != -1 or args.use_na_spans:
        with jsonlines.open("data/{}.parser.spanbert.jsonlines".format(args.dataset), mode='w') as w:
            for i, output in enumerate(mapped_outputs):
                w.write(output)
