import jsonlines
import re
from tqdm import tqdm


def convert_bert_word(word):
    """
    Convert bert token to regular word. Convert -LRB- and -RRB- tokens to regular parentheses
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
    adjusted_mentions = []
    for mention in mentions:
        if mention[0] >= sent_start and mention[1] <= sent_end:
            adjusted_start = subtoken_map[mention[0]] - subtoken_map[sent_start]
            adjusted_end = subtoken_map[mention[1]] - subtoken_map[sent_start]
            adjusted_mentions.append((adjusted_start, adjusted_end))
    return adjusted_mentions


if __name__ == "__main__":
    dataset = 'dev'
    outputs = []
    with jsonlines.open("data/{}.output.english.512.jsonlines".format(dataset)) as reader:
        for line in reader:
            outputs.append(line)

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
            if i != 0 and sentence_map[i-1] != sentence_map[i]:  # New sentence
                sent_so_far.append(convert_bert_word(''.join(word_so_far)))
                word_so_far = []
                mapped_outputs.append({'words': sent_so_far,
                                       'clusters': adjust_cluster_indices(clusters, subtoken_map, sentence_start_idx, i-1),
                                       'predicted_clusters': adjust_cluster_indices(preds, subtoken_map, sentence_start_idx, i-1),
                                       'top_mentions': adjust_top_mentions(top_mentions, subtoken_map, sentence_start_idx, i-1)})
                sent_so_far = []
                sentence_start_idx = i
            elif i != 0 and subtoken_map[i-1] != subtoken_map[i]:  # New word
                sent_so_far.append(convert_bert_word(''.join(word_so_far)))
                word_so_far = []

            word_so_far.append(subword)

    with jsonlines.open('data/{}.adjust_span_sents.jsonlines'.format(dataset), mode='w') as w:
        for output in mapped_outputs:
            w.write(output)

    with open('data/{}.spans'.format(dataset), mode='w') as w:
        for output in mapped_outputs:
            w.write(" ".join([str(idx) for span in output['clusters'] for idx in span]) + '\n')

    with open('data/{}.raw_tokens.sentences'.format(dataset), mode='w') as f:
        for output in mapped_outputs:
            f.write(' '.join(output['words']).strip() + '\n')
