from argparse import ArgumentParser
import csv
from tqdm import tqdm
import re

def remove_thingtalk_quotes(thingtalk):
    quote_values = []
    while True:
        # print('before: ', thingtalk)
        l1 = thingtalk.find('"')
        if l1 < 0:
            break
        l2 = thingtalk.find('"', l1+1)
        if l2 < 0:
            # ThingTalk code is not syntactic
            return thingtalk, None
        quote_values.append(thingtalk[l1+1: l2].strip())
        thingtalk = thingtalk[:l1] + '<temp>' + thingtalk[l2+1:]
        # print('after: ', thingtalk)
    thingtalk = thingtalk.replace('<temp>', '""')
    return thingtalk, quote_values


def main():
    parser = ArgumentParser()
    parser.add_argument('input', type=str,
                        help='The path to the input.')
    parser.add_argument('output', type=str,
                        help='The path to the output file.')
    parser.add_argument('--query_file', type=str,
                        help='The path to the file containing new queries.')
    parser.add_argument('--thingtalk_gold_file', type=str,
                        help='The path to the file containing the dataset with a correct thingtalk column.')
    parser.add_argument('--num_new_queries', type=int, default=1,
                        help='Number of new queries per old query. Valid if "--transformation replace_queries" is used.')
    parser.add_argument('--transformation', type=str, choices=['remove_thingtalk_quotes', 'replace_queries', 'remove_wrong_thingtalk', 'get_wrong_thingtalk', 'none'], default='none',
                        help='The type of transformation to apply.')
    parser.add_argument('--remove_duplicates', action='store_true',
                        help='Remove duplicate natural utterances. Note that this also removes cases where a natural utterance has multiple ThingTalk codes.')
    parser.add_argument('--output_columns', type=int, nargs='+', default=None,
                        help='The columns to write to output. By default, we output all columns.')
    parser.add_argument('--utterance_column', type=int, default=1,
                        help='The column index in the input file that contains the natural utterance')
    parser.add_argument('--thingtalk_column', type=int, default=2,
                        help='The column index in the input file that contains the ThingTalk code.')

    # These arguments are effective only if --transformation=get_wrong_thingtalk
    parser.add_argument('--remove_with_heuristics', action='store_true',
                        help='Remove examples if the values inside quotations in ThingTalk have changed or special words like NUMBER_0 cannot be found in TT anymore.')
    parser.add_argument('--replace_with_gold', action='store_true', help='Instead of the original ThingTalk, output what the parser said is gold.')


    args = parser.parse_args()

    if args.output_columns is None:
        # if args.transformation == 'remove_wrong_thingtalk':
            # args.output_columns = [0, 1, 2, 3, 4] # we add original utterance and ThingTalk as well
        args.output_columns = [0, 1, 2]

    with open(args.input, 'r') as input_file, open(args.output, 'w') as output_file:
        reader = csv.reader(input_file, delimiter='\t')
        if args.transformation == 'replace_queries':
            new_queries = []
            query_file = open(args.query_file, 'r')
            for q in query_file:
                new_queries.append(q.strip())
            new_query_count = 0
        if args.transformation in ['remove_wrong_thingtalk', 'get_wrong_thingtalk']:
            gold_thingtalks = []
            thingtalk_gold_file_reader = csv.reader(open(args.thingtalk_gold_file, 'r'), delimiter='\t')
            for q in thingtalk_gold_file_reader:
                gold_thingtalks.append(q[1].strip())
            gold_thingtalk_count = 0
            
        duplicate_count = 0
        heuristic_count = 0
        written_count = 0
        if args.remove_duplicates:
            seen_natural_utterances = set()
        for row in tqdm(reader, desc='Lines'):
            output_rows = []
            if args.transformation == 'remove_thingtalk_quotes':
                row[2], _ = remove_thingtalk_quotes(row[2])
            if args.transformation == 'remove_wrong_thingtalk':
                if row[2] == gold_thingtalks[gold_thingtalk_count]:
                    output_rows = [row]
                else:
                    output_rows = []
                gold_thingtalk_count += 1
            elif args.transformation == 'get_wrong_thingtalk':
                if row[2] != gold_thingtalks[gold_thingtalk_count]:
                    if args.remove_with_heuristics:
                        _, new_quote_values = remove_thingtalk_quotes(row[2])
                        _, gold_quote_values = remove_thingtalk_quotes(gold_thingtalks[gold_thingtalk_count])
                        if new_quote_values != gold_quote_values:
                            output_rows = []
                            heuristic_count += 1
                        else:
                            if args.replace_with_gold:
                                row[2] = gold_thingtalks[gold_thingtalk_count]
                            if set(re.findall('[A-Z]+_[0-9]', row[1])) != set(re.findall('[A-Z]+_[0-9]', row[2])):
                                output_rows = []
                                heuristic_count += 1
                            else:
                                output_rows = [row]
                    else:
                        if args.replace_with_gold:
                            row[2] = gold_thingtalks[gold_thingtalk_count]
                        output_rows = [row]
                else:
                    output_rows = []
                gold_thingtalk_count += 1
            elif args.transformation == 'replace_queries':
                for _ in range(args.num_new_queries):
                    row[1] = new_queries[new_query_count]
                    output_rows.append(row.copy())
                    new_query_count += 1
            else:
                assert args.transformation == 'none'
                # do basic clean-up because old generation code missed these
                row[1] = row[1].replace('<pad>', '')
                row[1] = re.sub('\s\s+', ' ', row[1])
                row[1] = row[1].strip()
                output_rows = [row]
            
            for o in output_rows:
                output_row = ""
                if args.remove_duplicates:
                    normalized_utterance = re.sub('\s+', '', o[args.utterance_column])
                    if normalized_utterance in seen_natural_utterances:
                        duplicate_count += 1
                        continue
                    else:
                        seen_natural_utterances.add(normalized_utterance)
                written_count += 1
                for i, column in enumerate(args.output_columns):
                    output_row += o[column]
                    if i < len(args.output_columns)-1:
                        output_row += '\t'
                output_file.write(output_row + '\n')
        print('Removed %d duplicate rows' % duplicate_count)
        print('Removed %d rows because the thingtalk quote did not satisfy heuristic rules' % heuristic_count)
        print('Output size is %d rows' % written_count)

if __name__ == '__main__':
    main()
