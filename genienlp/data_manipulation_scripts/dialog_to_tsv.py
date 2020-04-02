from argparse import ArgumentParser
import csv
from tqdm import tqdm
import re


def read_dialog_file(dialog_file, args):
    all_prompts = {}
    dialog_lines = []
    with open(dialog_file) as f:
        for l in f:
            dialog_lines.append(l.strip())

    idx = 0
    dialog_idx = 0
    while idx < len(dialog_lines):
        if dialog_lines[idx] == args.dialog_start:
            # new dialog started
            dialog_start_line_idx = idx
            idx += 1
            while idx < len(dialog_lines) and dialog_lines[idx] != args.dialog_start:
                idx += 1
            prompts = read_dialog(dialog_lines[dialog_start_line_idx: idx], args)
            all_prompts[dialog_idx] = prompts
            dialog_idx += 1
    return all_prompts

def read_dialog(lines: list, args):
    prompts = []
    agent_utterances = ['']
    previous_ats = []
    next_uts = []
    user_utterance = None
    # index = 0
    for idx, l in enumerate(lines):
        # if l.startswith(args.index_prefix):
            # index = l[len(args.index_prefix):].strip()
        if l.startswith(args.at_prefix):
            previous_ats.append(l[len(args.at_prefix):].strip())
        if l.startswith(args.ut_prefix):
            next_uts.append(l[len(args.ut_prefix):].strip())
            if idx == len(lines)-1 or not lines[idx+1].startswith(args.ut_prefix):
                # this is the last UT, so complete the example
                if len(previous_ats) == 0:
                    previous_ats.append('null')
                previous_ats = []
                next_uts = []
        if l.startswith(args.agent_prefix):
            agent_utterances.append(l[len(args.agent_prefix):].strip())
        if l.startswith(args.user_prefix):
            user_utterance = l[len(args.user_prefix):].strip()
            last_agent_utterance = agent_utterances[-1]
            prompts.append(((last_agent_utterance+' '+user_utterance).strip(), (last_agent_utterance+' ').strip()))

    return prompts

def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=str,
                        help='The path to the input file that has replaced parameters.')
    parser.add_argument('--dialog_file', type=str,
                        help='The path to the input file that has the original (unrepplaced) dialogs.')
    parser.add_argument('output', type=str,
                        help='The path to the output file that can be used for paraphrasing.')
    parser.add_argument('--dialog_start', type=str, default='====', help='')
    parser.add_argument('--index_prefix', type=str, default='#', help='')
    parser.add_argument('--user_prefix', type=str, default='U:', help='')
    parser.add_argument('--agent_prefix', type=str, default='A:', help='')
    parser.add_argument('--at_prefix', type=str, default='AT:', help='')
    parser.add_argument('--ut_prefix', type=str, default='UT:', help='')

    args = parser.parse_args()

    all_prompts = read_dialog_file(args.dialog_file, args)
    
    with open(args.input) as input:
        reader = csv.reader(input, delimiter='\t')
        with open(args.output, 'w') as output:
            for row in reader:
                assert row[0].startswith('RS')
                # print(row[0])
                slash = row[0].index('/')
                dash = row[0].index('-')
                dialog_index = int(row[0][2:slash])
                utterance_index = int(row[0][slash+1:dash])
                # print(row[2])
                # print(all_prompts[dialog_index][utterance_index][0])
                output.write(all_prompts[dialog_index][utterance_index][0]+'\t'+all_prompts[dialog_index][utterance_index][1]+'\n')


if __name__ == '__main__':
    main()
