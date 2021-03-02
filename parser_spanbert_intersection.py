import jsonlines
import sys
from score_parser_spans import load_spans
import random

if __name__ == "__main__":
    dataset = 'dev'
    outputs = []
    with jsonlines.open("data/{}.parser.spanbert.jsonlines".format(dataset), 'r') as reader:
        for line in reader:
            outputs.append(line)

    counts = {
        'gold': 0,
        'spanbert': 0,
        'top_m': 0,
        'parser': 0,
        'sb_p_union': 0,  # SpanBert Parser Union
        'tm_p_union': 0,  # SpanBert (top mentions) Parser Union
        'sb_p_inter': 0,  # SpanBert Parser Intersection
        'tm_p_inter': 0,  # SpanBert (top mentions) Parser Intersection
        'sb_rand_rem': 0, # Randomly removing mentions from SpanBert
        'tm_rand_rem': 0, # Randomly removing mentions from Top Mentions model
    }

    correct_counts = {
        'spanbert': 0,
        'top_m': 0,
        'parser': 0,
        'sb_p_union': 0,  # SpanBert Parser Union
        'tm_p_union': 0,  # SpanBert (top mentions) Parser Union
        'sb_p_inter': 0,  # SpanBert Parser Intersection
        'tm_p_inter': 0,  # SpanBert (top mentions) Parser Intersection
        'sb_rand_rem': 0,  # Randomly removing mentions from SpanBert
        'tm_rand_rem': 0,  # Randomly removing mentions from Top Mentions model
    }

    for output in outputs:
        gold = set([tuple(x) for x in output['clusters']])
        models = {
            'spanbert': set([tuple(x) for x in output['predicted_clusters']]),
            'top_m': set([tuple(x) for x in output['top_mentions']]),
            'parser': set([tuple(x) for x in output['parser_clusters']])
        }

        # Union
        models['sb_p_union'] = models['spanbert'].union(models['parser'])
        models['tm_p_union'] = models['top_m'].union(models['parser'])

        # Intersection
        models['sb_p_inter'] = models['spanbert'].intersection(models['parser'])
        models['tm_p_inter'] = models['top_m'].intersection(models['parser'])

        # Random removal
        models['sb_rand_rem'] = random.sample(models['spanbert'], len(models['sb_p_inter']))
        models['tm_rand_rem'] = random.sample(models['top_m'], len(models['tm_p_inter']))

        # Score all models
        counts['gold'] += len(gold)
        for model_key in models.keys():
            counts[model_key] += len(models[model_key])
            correct_counts += len(models[model_key].intersection(gold))

    # Reporting
    print("RECALL")
    for model_key in correct_counts.keys():
        recall = correct_counts[model_key] / counts['gold']
        print("Model={}, Correct Count={}, Recall={}".format(model_key, correct_counts[model_key], recall))

    print("PRECISION")
    for model_key in correct_counts.keys():
        precision = correct_counts[model_key] / counts[model_key]
        print("Model={}, Correct Count={}, Counts={}, Precision={}".format(model_key, correct_counts[model_key], counts[model_key], precision))
