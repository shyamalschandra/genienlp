from argparse import ArgumentParser
import csv
from tqdm import tqdm
import re


def read_dialog_file(file_path, args):
    all_prompts = []
    all_examples = []
    with open(file_path) as f:
        lines = []
        for l in f:
            lines.append(l.strip())
    idx = 0
    while idx < len(lines):
        if lines[idx] == args.dialog_start:
            # new dialog started
            dialog_start_line_idx = idx
            idx += 1
            while idx < len(lines) and lines[idx] != args.dialog_start:
                idx += 1
            prompts, examples = read_dialog(lines[dialog_start_line_idx: idx], args)
            all_prompts.extend(prompts)
            all_examples.extend(examples)
    return all_prompts, all_examples

def read_dialog(lines: list, args):
    prompts = []
    examples = []
    agent_utterances = ['']
    previous_ats = []
    next_uts = []
    user_utterance = None
    for idx, l in enumerate(lines):
        if l.startswith(args.at_prefix):
            previous_ats.append(l[len(args.at_prefix):].strip())
        if l.startswith(args.ut_prefix):
            next_uts.append(l[len(args.ut_prefix):].strip())
            if idx == len(lines)-1 or not lines[idx+1].startswith(args.ut_prefix):
                # this is the last UT, so complete the example
                if len(previous_ats) == 0:
                    previous_ats.append('null')
                examples.append((' '.join(previous_ats), user_utterance, ' '.join(next_uts)))
                previous_ats = []
                next_uts = []
        if l.startswith(args.agent_prefix):
            agent_utterances.append(l[len(args.agent_prefix):].strip())
        if l.startswith(args.user_prefix):
            user_utterance = l[len(args.user_prefix):].strip()
            last_agent_utterance = agent_utterances[-1]
            prompts.append(((last_agent_utterance+' '+user_utterance).strip(), (last_agent_utterance+' ').strip()))

    return prompts, examples

def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=str,
                        help='The path to the input file.')
    parser.add_argument('--prompt_output', type=str,
                        help='The path to the output file that can be used for paraphrasing.')
    parser.add_argument('--example_output', type=str,
                        help='The path to the output file that can be used to train the parser.')
    parser.add_argument('--dialog_start', type=str, default='====', help='')
    parser.add_argument('--user_prefix', type=str, default='U:', help='')
    parser.add_argument('--agent_prefix', type=str, default='A:', help='')
    parser.add_argument('--at_prefix', type=str, default='AT:', help='')
    parser.add_argument('--ut_prefix', type=str, default='UT:', help='')

    args = parser.parse_args()

    all_prompts, all_examples = read_dialog_file(args.input, args)
    # print(all_prompts)
    with open(args.prompt_output, 'w') as f:
        for prompt in all_prompts:
            f.write(prompt[0]+'\t'+prompt[1]+'\n')

    with open(args.example_output, 'w') as f:
        for idx, example in enumerate(all_examples):
            f.write(str(idx)+'\t'+example[0]+'\t'+example[1]+'\t'+example[2]+'\n')
            

if __name__ == '__main__':
    main()
