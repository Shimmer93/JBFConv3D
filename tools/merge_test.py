from mmengine import load, dump
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='merge any number of pkl results and calculate the accuracy')
    parser.add_argument('results', type=str, nargs='+', help='result pkl files')
    parser.add_argument('--out', type=str, required=True, help='output pkl file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    result_files = args.results
    out_file = args.out

    if os.path.exists(out_file):
        print(f'Output file {out_file} already exists.')
        merged_results = load(out_file)
    else:
        all_results = [load(f) for f in result_files]
        merged_results = []
        for results in zip(*all_results):
            r = {}
            r['pred_score'] = sum([res['pred_score'] for res in results]) / len(results)
            r['pred_label'] = r['pred_score'].argmax()
            r['gt_label'] = results[0]['gt_label']
            merged_results.append(r)
        
        dump(merged_results, out_file)
        print(f'Merged {len(result_files)} result files into {out_file}')
    
    acc = (sum([r['pred_label'] == r['gt_label'] for r in merged_results]) / len(merged_results)).item()
    print(f'Accuracy: {acc:.4f}')