import jsonlines
import re
from tqdm import tqdm
import os
from score_parser_spans import load_spans
import argparse
import stanza
import logging


def get_config():
    """
    Configuration for which token processing will be computed.
    """
    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("--dataset", type=str, default='dev')
    config_parser.add_argument("--ner", action="store_true", help="use NER to group NE tokens")
    config_parser.add_argument("--punc", action="store_true", help="join all hyphenated words")
    config_parser.add_argument("--parser_preds", type=int, default=0, help="attach parser preds with top-k categories")
    config_parser.add_argument("--na_file", type=str, default="", help="attach mentions not captured by parser")
    config_parser.add_argument("--reverse_map", action="store_true", help="create maps from parse tree to bert indices")
    config_parser.add_argument("--max_segment_len", type=str, default="384", help="max segment length")
    config_parser.add_argument("--use_gpu", action='store_true')
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


def add_na_spans(na_filename, mapped_outputs):
    """
    Adds mentions that do not correspond to any node in the tree to the output dictionary.
    """
    na_file = "data/{}".format(na_filename)
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


def create_grouped_word_list(words, group_span_indices, join_string):
    """Group together words with join_string string and return updated token list."""
    adjusted_words = []
    curr_group = []
    group_idx = 0
    curr_group_start_idx, curr_group_end_idx = group_span_indices[group_idx]
    for i, token in enumerate(words):
        if curr_group_start_idx <= i <= curr_group_end_idx:
            curr_group.append(token)
            if i == curr_group_end_idx:
                adjusted_words.append(join_string.join(curr_group))

                # Update various indices
                curr_group = []
                group_idx += 1
                if group_idx < len(group_span_indices):
                    curr_group_start_idx, curr_group_end_idx = group_span_indices[group_idx]  # Track next group
                else:
                    curr_group_start_idx, curr_group_end_idx = len(words), len(words)  # Don't enter this block anymore
        else:
            adjusted_words.append(token)
    return adjusted_words


def valid_mapping(mention_start, mention_end, group_indices):
    """Determine if the mention can be mapped under merging rules."""
    for group_start, group_end in group_indices:
        if mention_start == group_start and mention_end == group_end:  # Exact match
            return True
        elif group_start <= mention_start <= group_end and group_start <= mention_end <= group_end:  # Partial or full nested
            return False
        elif mention_start < group_start <= mention_end < group_end:  # Partial overlap, left
            return False
        elif group_start < mention_start <= group_end < mention_end:  # Partial overlap, right
            return False
    return True


def adjust_grouped_mention_indices(mention_indices, group_indices, subtoken_map):
    """Adjusts mention indices after grouping certain indices into single tokens (e.g. named entities or hyphenated words).
    Returns the adjusted indices, and mention indices that cannot be mapped due to the NER changes."""
    adjusted_mention_indices = []
    error_indices = []
    for mention_start, mention_end in mention_indices:
        if valid_mapping(mention_start, mention_end, group_indices):
            adjusted_mention_indices.append((subtoken_map[mention_start], subtoken_map[mention_end]))
        else:
            error_indices.append((mention_start, mention_end))
    return adjusted_mention_indices, error_indices


def create_subtoken_map(tokens_len, indices):
    """
    Creates subtoken map by grouping specified indices.
    """
    if len(indices) == 0:
        return list(range(tokens_len))  # No entities, nothing to do

    k = 0  # Tracks new tokens with named entities grouped
    j = 0  # Tracks which named entity we are processing
    subtoken_map = {}
    sorted_indices = sorted(indices, key=lambda x: x[0])
    curr_start, curr_end = sorted_indices[j]
    for i in range(tokens_len):
        if i > curr_end and j < len(sorted_indices) - 1:  # Move up to next named entity
            j += 1
            curr_start, curr_end = sorted_indices[j]

        if curr_start <= i < curr_end:
            subtoken_map[i] = k
        else:
            subtoken_map[i] = k
            k += 1
    return list(subtoken_map.values())


def find_ner_indices(sents, ner_model):
    """
    Run NER model over a document and return list of list of tuples corresponding to entity index ranges for each sent.
    """
    sents = "\n".join(sents)
    doc = ner_model(sents)
    entity_indices = []
    for sent in doc.sentences:
        sent_entity_indices = []
        ent_begin_idx = -1
        for i, token in enumerate(sent.tokens):
            # Ignore S and O tokens, since they will not be modified.
            if token.ner[0] == "B":
                ent_begin_idx = i
            elif token.ner[0] == "E":
                sent_entity_indices.append((ent_begin_idx, i))
        entity_indices.append(sent_entity_indices)
    return entity_indices


def adjust_with_ner(mapped_outputs, use_gpu):
    """
    Create new token lists with NE grouped together, adjust mention indices accordingly and compute resulting MD errors.
    """
    stanza.download('en')
    nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', tokenize_pretokenized=True, use_gpu=use_gpu)
    sents_so_far = []
    entity_indices = []
    curr_doc_key = mapped_outputs[0]['doc_key']
    logging.info("Running NER.")
    for output in tqdm(mapped_outputs):
        if output['doc_key'] != curr_doc_key:  # After reaching the end of document, we run NER.
            sent_entity_indices = find_ner_indices(sents_so_far, nlp)
            entity_indices += sent_entity_indices
            sents_so_far = []

        sents_so_far.append(" ".join(output['words']))
        curr_doc_key = output['doc_key']
    sent_entity_indices = find_ner_indices(sents_so_far, nlp)
    entity_indices += sent_entity_indices
    logging.info("Entity indices: {}, Mapped Outputs: {}".format(len(entity_indices), len(mapped_outputs)))
    assert len(entity_indices) == len(mapped_outputs)

    logging.info("Formatting output dictionary and adjusting indices")
    for i, (output, sent_entity_indices) in enumerate(zip(mapped_outputs, entity_indices)):
        if sent_entity_indices:
            output['ne_adjusted_words'] = create_grouped_word_list(output['words'], sent_entity_indices, "<NE>")
            subtoken_map = create_subtoken_map(len(output['words']), sent_entity_indices)
            adj_mention_idx, error_mention_idx = adjust_grouped_mention_indices(output['clusters'], sent_entity_indices, subtoken_map)
            output['ne_adjusted_mention_indices'] = adj_mention_idx
            output['ne_mention_error_indices'] = error_mention_idx
        else:  # No named entities in this sentence.
            output['ne_adjusted_words'] = output['words']
            output['ne_adjusted_mention_indices'] = output['clusters']
            output['ne_mention_error_indices'] = []
    return mapped_outputs


def merge_index_boundaries(indices):
    """
    Auxiliary function to merge the boundaries of adjacent spans into continuous ones.
    """
    merged_indices = []
    j = 0
    while j < len(indices):
        curr_start, curr_end = indices[j]
        continue_merge = j < len(indices) - 1 and curr_end == indices[j + 1][0]
        while continue_merge:
            curr_end = indices[j + 1][1]
            j += 1
            continue_merge = j < len(indices) - 1 and curr_end == indices[j + 1][0]
        j += 1
        merged_indices.append((curr_start, curr_end))
    return merged_indices


def find_hyphenated_word_indices(words):
    """
    Compute boundary indices for hyphenated words.
    """
    hyphen_simple_boundaries = [(i - 1, i + 1) for i, x in enumerate(words) if x == '-' and 0 < i < len(words) - 1]
    return merge_index_boundaries(hyphen_simple_boundaries)


def find_quote_indices(words):
    """
    If start quote, group quote with word proceeding it.
    If end quote, group quote with word preceding it.
    Also handle directionless '"' quotes. NOTE: the method assumes there are an even number of '"' quotes.
    If this is not the case, this method may not be correct.
    Return list of tuples indicating groups indices, i.e. the quote index and its predecessor/successor
    """
    quote_indices = []
    start_quote = True
    for i, word in enumerate(words):
        if word == '``':
            quote_indices.append((i, i+1))
        elif word == "''":
            quote_indices.append((i-1, i))
        elif word == '"':
            if i == len(words) - 1:
                quote_indices.append((i-1, i))
            elif i == 0:
                quote_indices.append((i, i+1))
            else:
                quote_indices.append((i-1, i) if start_quote else (i, i+1))
            start_quote = not start_quote
    return merge_index_boundaries(quote_indices)


def adjust_punctuation(mapped_outputs):
    """
    Create new token lists with hyphenated words joined, strip out quotes and adjust mention indices for these cases.
    """
    logging.info("Adjusting punctuation cases.")
    for output in tqdm(mapped_outputs):
        words = output['words']

        # Quotation adjustments: strip out
        quote_syms = ["``", "''", '"']
        if any([quote_sym in words for quote_sym in quote_syms]):
            quote_indices = find_quote_indices(words)
            subtoken_map = create_subtoken_map(len(words), quote_indices)
            adj_mention_idx, error_mention_idx = adjust_grouped_mention_indices(output['clusters'], quote_indices, subtoken_map)
            output['punc_adjusted_mention_indices'] = adj_mention_idx
            output['punc_mention_error_indices'] = []  # These should be recoverable in post-processing
            adjusted_words = [word for i, word in enumerate(words) if word not in quote_syms]
        else:
            output['punc_adjusted_mention_indices'] = output['clusters']
            output['punc_mention_error_indices'] = []
            adjusted_words = words.copy()

        # Hyphen adjustments: join together.
        hyphenated_word_indices = find_hyphenated_word_indices(adjusted_words)
        if hyphenated_word_indices:
            subtoken_map = create_subtoken_map(len(adjusted_words), hyphenated_word_indices)
            adj_mention_idx, error_mention_idx = adjust_grouped_mention_indices(output['punc_adjusted_mention_indices'], hyphenated_word_indices, subtoken_map)
            output['punc_adjusted_mention_indices'] = adj_mention_idx
            output['punc_mention_error_indices'] = error_mention_idx
            output['punc_adjusted_words'] = create_grouped_word_list(adjusted_words, hyphenated_word_indices, "")
        else:
            output['punc_adjusted_words'] = adjusted_words

    return mapped_outputs


def convert_bert_tokens(outputs):
    """
    Converts BERT tokens into a readable format for the parser, i.e. using Penn Treebank tokenization scheme.
    Does the heavy lifting for this script.
    """
    logging.info("Adjusting BERT indices to align with Penn Treebank.")
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


def combine_subtoken_maps(*subtoken_maps):
    """
    Create a single subtoken map from BERT to parser indices.
    Auxiliary function that facilities easier reverse mapping.
    """
    combo_map_len = len(subtoken_maps[0])
    combo_map = []
    for i in range(combo_map_len):
        idx = i
        for submap in subtoken_maps:
            idx = submap[idx]
        combo_map.append(idx)
    return combo_map


def reverse_subtoken_map(subtoken_map):
    """
    Creates a map: idx --> (left_span_boundary, right_span_boundary) that effectively reverse the map created in
    create_subtoken_map.
    """
    reverse_map = {}
    for idx, mapped_val in enumerate(subtoken_map):
        if mapped_val in reverse_map:
            reverse_map[mapped_val] = (min(idx, reverse_map[mapped_val][0]), max(idx, reverse_map[mapped_val][1]))
        else:
            reverse_map[mapped_val] = (idx, idx)
    return list(reverse_map.values())


def create_reverse_map(outputs):
    """
    Creates a reverse map from parse tree indices to BERT indices for each document in the dataset.
    """
    tree_preprocess_docs = []
    for doc in outputs:
        # Create combined map
        bert_subtoken_map = doc['subtoken_map']
        penn_words = doc['tokens']
        no_quote_words = create_grouped_word_list(penn_words, find_quote_indices(penn_words), "")
        # parser_words = create_grouped_word_list(no_quote_words, find_hyphenated_word_indices(no_quote_words), "")

        noquote_map = create_subtoken_map(len(penn_words), find_quote_indices(penn_words))
        nohyphen_map = create_subtoken_map(len(no_quote_words), find_hyphenated_word_indices(no_quote_words))

        combined_map = combine_subtoken_maps(bert_subtoken_map, noquote_map, nohyphen_map)

        # Create reverse map
        reverse_map = reverse_subtoken_map(combined_map)

        tree_data = {
            'doc_key': doc['doc_key'],
            'bert': [tok for segment in doc['sentences'] for tok in segment],
            'sentence_map': doc['sentence_map'],
            'reverse_map': reverse_map
        }
        tree_preprocess_docs.append(tree_data)
    return tree_preprocess_docs


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    args = get_config()
    outputs = []
    with jsonlines.open("data/{}.output.english.{}.jsonlines".format(args.dataset, args.max_segment_len)) as reader:
        for line in reader:
            outputs.append(line)

    mapped_outputs = convert_bert_tokens(outputs)
    file_adj_prefix = ""
    if args.ner:
        mapped_outputs = adjust_with_ner(mapped_outputs, args.use_gpu)
        file_adj_prefix += "ner."
    if args.punc:
        mapped_outputs = adjust_punctuation(mapped_outputs)
        file_adj_prefix += "punc."

    if args.reverse_map:
        reverse_map = create_reverse_map(outputs)
        reverse_map_name = "data/{}.reverse_map.{}.jsonlines".format(args.dataset, args.max_segment_len)
        with jsonlines.open(reverse_map_name, mode='w') as w:
            for output in reverse_map:
                w.write(output)

    with jsonlines.open('data/{}.{}adjust_span_sents.jsonlines'.format(args.dataset, file_adj_prefix), mode='w') as w:
        for output in mapped_outputs:
            w.write(output)

    with open('data/{}.{}spans'.format(args.dataset, file_adj_prefix), mode='w') as w:
        for output in mapped_outputs:
            if args.ner:
                to_write = " ".join([str(idx) for span in output['ne_adjusted_mention_indices'] for idx in span])
            elif args.punc:
                to_write = " ".join([str(idx) for span in output['punc_adjusted_mention_indices'] for idx in span])
            else:
                to_write = " ".join([str(idx) for span in output['clusters'] for idx in span])
            w.write(to_write + '\n')

    with open('data/{}.{}raw_tokens.sentences'.format(args.dataset, file_adj_prefix), mode='w') as f:
        for output in mapped_outputs:
            if args.ner:
                to_write = ' '.join(output['ne_adjusted_words']).strip()
            elif args.punc:
                to_write = ' '.join(output['punc_adjusted_words']).strip()
            else:
                to_write = ' '.join(output['words']).strip()
            f.write(to_write + '\n')

    # Add on parser predictions
    if args.parser_preds:
        add_parser_preds(args, mapped_outputs)

    if args.na_file:
        add_na_spans(args.na_file, mapped_outputs)

    if args.parser_preds != -1 or args.na_file:
        with jsonlines.open("data/{}.parser.spanbert.jsonlines".format(args.dataset), mode='w') as w:
            for i, output in enumerate(mapped_outputs):
                w.write(output)
