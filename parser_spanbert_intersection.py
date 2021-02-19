import jsonlines
import sys
from score_parser_spans import load_spans


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
                parser_spans.append(load_spans(line))
            else:
                parser_spans.append(set())

    num_spanbert_parser_correct = 0
    num_spanbert_top_mention_parser_correct = 0
    num_spanbert_correct = 0
    num_spanbert_top_mention_correct = 0
    num_spanbert_intersection_correct = 0
    num_spanbert_top_mention_intersection_correct = 0
    num_gold = 0
    num_parse = 0
    num_spanbert = 0
    num_spanbert_top_mention = 0
    num_spanbert_intersection = 0
    num_spanbert_top_mention_intersection = 0
    for gold_span, spanbert_span, spanbert_top_mention, parser_span in zip(gold_spans, spanbert_spans, spanbert_top_mentions, parser_spans):
        # Union
        spanbert_parser_union = spanbert_span.union(parser_span)
        spanbert_top_mention_parser_union = spanbert_top_mention.union(parser_span)
        spanbert_parser_union_correct = gold_span.intersection(spanbert_parser_union)
        spanbert_top_mention_parser_union_correct = gold_span.intersection(spanbert_top_mention_parser_union)
        num_spanbert_parser_correct += len(spanbert_parser_union_correct)
        num_spanbert_top_mention_parser_correct += len(spanbert_top_mention_parser_union_correct)

        # Intersection
        spanbert_parser_intersection = spanbert_span.intersection(parser_span)
        spanbert_top_mention_parser_intersection = spanbert_top_mention.intersection(parser_span)
        spanbert_parser_intersection_correct = gold_span.intersection(spanbert_parser_intersection)
        spanbert_top_mention_parser_intersection_correct = gold_span.intersection(spanbert_top_mention_parser_intersection)
        num_spanbert_intersection_correct += len(spanbert_parser_intersection_correct)
        num_spanbert_top_mention_intersection_correct += len(spanbert_top_mention_parser_intersection_correct)

        num_spanbert_intersection += len(spanbert_parser_intersection)
        num_spanbert_top_mention_intersection += len(spanbert_top_mention_parser_intersection)

        spanbert_correct = gold_span.intersection(spanbert_span)
        spanbert_top_mention_correct = gold_span.intersection(spanbert_top_mention)


        num_spanbert_correct += len(spanbert_correct)
        num_spanbert_top_mention_correct += len(spanbert_top_mention_correct)
        num_gold += len(gold_span)
        num_parse += len(parser_span)
        num_spanbert += len(spanbert_span)
        num_spanbert_top_mention += len(spanbert_top_mention)

    print("# spanbert-parser union correct: {}, ".format(num_spanbert_parser_correct))
    print("# spanbert (top mentions)-parser union correct: {},".format(num_spanbert_top_mention_parser_correct))
    print("# spanbert correct: {}".format(num_spanbert_correct))
    print("# spanbert (top mentions) correct: {}".format(num_spanbert_top_mention_correct))
    print("# gold: {}".format(num_gold))

    print("# spanbert-parser intersection correct: {}".format(num_spanbert_intersection_correct))
    print("# spanbert (top mentions)-parser intersection correct: {}".format(num_spanbert_top_mention_intersection_correct))
    print("# spanbert-parser intersection: {}".format(num_spanbert_intersection))
    print("# spanbert (top mentions)-parser intersection: {}".format(num_spanbert_top_mention_intersection))
    print("# spanbert: {}".format(num_spanbert))
    print("# spanbert (top mentions): {}".format(num_spanbert_top_mention))
