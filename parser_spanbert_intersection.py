import jsonlines
import sys


if __name__ == "__main__":
    dataset = 'dev'

    # Load gold spans
    gold_spans = []
    spanbert_spans = []
    spanbert_top_mentions = []
    with jsonlines.open("data/{}.adjust_span_sents.jsonlines".format(dataset)) as reader:
        for line in reader:
            gold_spans.append(set([tuple(x) for x in line['clusters']]))
            spanbert_spans.append(set([tuple(x) for x in line['predicted_clusters']]))
            spanbert_top_mentions.append(set([tuple(x) for x in line['top_mentions']]))

    # Load pred spans
    parser_spans = []
    with open("data/{}_preds/{}.{}.preds".format(dataset, dataset, int(sys.argv[1])), 'r') as f:
        for line in f:
            if line.strip():
                parser_spans.append(set(eval(line.strip()[5:-1])))
            else:
                parser_spans.append(set())

    num_spanbert_parser_intersect = 0
    num_spanbert_top_mention_parser_intersect = 0
    num_parser_correct = 0
    num_gold = 0
    for gold_span, spanbert_span, spanbert_top_mention, parser_span in zip(gold_spans, spanbert_spans, spanbert_top_mentions, parser_spans):
        correct_spanbert_spans = gold_span.intersection(spanbert_span)
        correct_spanbert_top_mentions = gold_span.intersection(spanbert_top_mention)
        correct_parser_spans = gold_span.intersection(parser_span)

        spanbert_parser_intersection = correct_spanbert_spans.intersection(correct_parser_spans)
        spanbert_top_mention_parser_intersections = correct_spanbert_top_mentions.intersection(correct_parser_spans)

        num_spanbert_parser_intersect += len(spanbert_parser_intersection)
        num_spanbert_top_mention_parser_intersect += len(spanbert_top_mention_parser_intersections)
        num_parser_correct += len(correct_parser_spans)
        num_gold += len(gold_span)

    print("# spanbert-parser correct intersection: {}, ".format(num_spanbert_parser_intersect))
    print("# spanbert (top mentions)-parser correct intersection: {},".format(num_spanbert_top_mention_parser_intersect))
    print("# parser correct: {}".format(num_parser_correct))
    print("# gold: {}".format(num_gold))
