import jsonlines
import sys


if __name__ == "__main__":
    dataset = 'dev'

    # Load gold spans
    gold_spans = []
    with jsonlines.open("data/{}.adjust_span_sents.jsonlines".format(dataset)) as reader:
        for line in reader:
            gold_spans.append(set([tuple(x) for x in line['spans']]))

    # Load pred spans
    pred_spans = []
    with open("data/{}_preds/{}.{}.preds".format(dataset, dataset, int(sys.argv[1])), 'r') as f:
        for line in f:
            if line.strip():
                pred_spans.append(set(eval(line.strip()[5:-1])))
            else:
                pred_spans.append(set())

    num_correct = 0
    num_gold = 0
    num_pred = 0
    for gold_span, pred_span in zip(gold_spans, pred_spans):
        correct_spans = gold_span.intersection(pred_span)
        num_correct += len(correct_spans)
        num_gold += len(gold_span)
        num_pred += len(pred_span)

    precision = num_correct / num_pred
    recall = num_correct / num_gold
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0.0 else 0.0
    print("{}: precision={}, recall={}, f1={}".format(int(sys.argv[1]), precision, recall, f1))
