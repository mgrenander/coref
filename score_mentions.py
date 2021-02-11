import jsonlines
import numpy as np
import argparse


def convert_mention(output, mention):
    start = output['subtoken_map'][mention[0]]
    end = output['subtoken_map'][mention[1]] + 1
    nmention = (start, end)
    mtext = ''.join(' '.join(comb_text[mention[0]:mention[1]+1]).split(" ##"))
    return (nmention, mtext)


def MD_recall_precision_f1(gold_spans, pred_spans):
    gold_set = set(gold_spans)
    pred_set = set(pred_spans)
    intersect = gold_set.intersection(pred_set)

    if len(pred_set) == 0:
        precision = 0.0
    else:
        precision = len(intersect) / len(pred_set)
    recall = len(intersect) / len(gold_set)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2*precision*recall) / (precision + recall)
    return {'precision': precision, 'recall': recall, 'f1': f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dev')
    parser.add_argument('--pred_key', type=str, help='predicted_clusters or top_spans')
    args = parser.parse_args()

    dataset = args.dataset
    pred_key = args.pred_key
    outputs = []
    with jsonlines.open("data/{}.output.english.512.jsonlines".format(dataset)) as reader:
        for line in reader:
            outputs.append(line)

    # scored_outputs = []
    gold_len = 0
    pred_len = 0
    correct_preds_len = 0
    for output in outputs:

        # stats = {}
        # comb_text = [word for sentence in output['sentences'] for word in sentence]
        # stats['comb_text'] = comb_text
        gold_set = set() if not output['clusters'] else set([tuple(span) for cluster in output['clusters'] for span in cluster])
        preds_set = set() if not output[pred_key] else set([tuple(span) for cluster in output[pred_key] for span in cluster])
        correct_preds = gold_set.intersection(preds_set)
        gold_len += len(gold_set)
        pred_len += len(preds_set)
        correct_preds_len += len(correct_preds)

        # stats['pred_cluster_text'] = []
        # stats['pred_cluster_spans'] = []
        # for cluster in output['predicted_clusters']:
        #     cluster_text = []
        #     for mention in cluster:
        #         cluster_text.append(convert_mention(output, mention))
        #         stats['pred_cluster_spans'].append(tuple(mention))
        #     stats['pred_cluster_text'].append(cluster_text)
        #
        # # Gold spans
        # stats['gold_cluster_text'] = []
        # stats['gold_cluster_spans'] = []
        # for cluster in output['clusters']:
        #     cluster_text = []
        #     for mention in cluster:
        #         cluster_text.append(convert_mention(output, mention))
        #         stats['gold_cluster_spans'].append(tuple(mention))
        #     stats['gold_cluster_text'].append(cluster_text)
        #
        # # Top Spans
        # stats['pred_mention_text'] = []
        # stats['pred_mention_spans'] = []
        # for mention in output['top_spans']:
        #     stats['pred_mention_spans'].append(tuple(mention))
        #     stats['pred_mention_text'].append(convert_mention(output, mention))
        #
        # stats['cluster_scores'] = MD_recall_precision_f1(stats['gold_cluster_spans'], stats['pred_cluster_spans'])
        # stats['span_scores'] = MD_recall_precision_f1(stats['gold_cluster_spans'], stats['pred_mention_spans'])

        # scored_outputs.append(stats)

    recall = correct_preds_len / gold_len
    precision = 0.0 if pred_len == 0 else correct_preds_len / pred_len
    f1 = 0.0 if recall + precision == 0.0 else (2 * precision * recall) / (precision + recall)
    print("SpanBert MD counts using {}: # gold spans={}, # pred spans={}, # correct={}".format(pred_key, gold_len, pred_len, correct_preds_len))
    print("SpanBert MD stats using {}: recall={}, precision={}, f1={}".format(pred_key, recall, precision, f1))

    # Calculate macro-average and print. Don't add to stats since it will make serialization ugly
    # for measure in ['cluster_scores', 'span_scores']:
    #     avg_recall = np.mean([x[measure]['recall'] for x in scored_outputs])
    #     avg_precision = np.mean([x[measure]['precision'] for x in scored_outputs])
    #     avg_f1 = np.mean([x[measure]['f1'] for x in scored_outputs])
    #     print("{}: r={}, p={}, f1={}".format(measure, avg_recall, avg_precision, avg_f1))
    #
    # # Write to file
    # with jsonlines.open('data/scored.test.output.jsonlines', mode='w') as w:
    #     w.write(scored_outputs)
