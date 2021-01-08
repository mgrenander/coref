import jsonlines
import numpy as np


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
    outputs = []
    with jsonlines.open("data/test.output.english.512.jsonlines") as reader:
        for line in reader:
            outputs.append(line)

    scored_outputs = []
    for output in outputs:
        if not output['clusters']:
            continue

        stats = {}
        comb_text = [word for sentence in output['sentences'] for word in sentence]
        stats['comb_text'] = comb_text

        # Clusters
        stats['pred_cluster_text'] = []
        stats['pred_cluster_spans'] = []
        for cluster in output['predicted_clusters']:
            cluster_text = []
            for mention in cluster:
                cluster_text.append(convert_mention(output, mention))
                stats['pred_cluster_spans'].append(tuple(mention))
            stats['pred_cluster_text'].append(cluster_text)

        # Gold spans
        stats['gold_cluster_text'] = []
        stats['gold_cluster_spans'] = []
        for cluster in output['clusters']:
            cluster_text = []
            for mention in cluster:
                cluster_text.append(convert_mention(output, mention))
                stats['gold_cluster_spans'].append(tuple(mention))
            stats['gold_cluster_text'].append(cluster_text)

        # Top Spans
        stats['pred_mention_text'] = []
        stats['pred_mention_spans'] = []
        for mention in output['top_spans']:
            stats['pred_mention_spans'].append(tuple(mention))
            stats['pred_mention_text'].append(convert_mention(output, mention))

        stats['cluster_scores'] = MD_recall_precision_f1(stats['gold_cluster_spans'], stats['pred_cluster_spans'])
        stats['span_scores'] = MD_recall_precision_f1(stats['gold_cluster_spans'], stats['pred_mention_spans'])

        scored_outputs.append(stats)

    # Calculate macro-average and print. Don't add to stats since it will make serialization ugly
    for measure in ['cluster_scores', 'span_scores']:
        avg_recall = np.mean([x[measure]['recall'] for x in scored_outputs])
        avg_precision = np.mean([x[measure]['precision'] for x in scored_outputs])
        avg_f1 = [x[measure]['f1'] for x in scored_outputs]
        print("{}: r={}, p={}, f1={}".format(measure, avg_recall, avg_precision, avg_f1))

    # Write to file
    with jsonlines.open('data/scored.test.output.jsonlines', mode='w') as w:
        w.write(scored_outputs)
