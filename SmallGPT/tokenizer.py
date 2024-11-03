import regex as re
import argparse 
import json


class GPT4Tokenizer():
    def __init__(self, path='vocab.json', pattern=None):
        self.vocab = {idx : bytes([idx]) for idx in range(256)}
        self.merges = dict()
        self.pattern = pattern if pattern else r"\p{L}+|\p{Z}+|\p{N}+|[\p{P}&&[^.]]"
        self.splitby = re.compile(self.pattern)
        self.path = path


    def train(self, text, vocab_size):

        assert vocab_size >= 256

        num_merges = vocab_size - 256

        text_splitted = re.findall(self.splitby, text)

        ids = [list(ch.encode("utf-8")) for ch in text_splitted]

        for i in range(num_merges):
            stats = {}
            for _ in ids:
                self.get_pairs(_, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [self.merge(chunk_ids, pair, idx) for chunk_ids in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.save_vocab_and_merges(self.path)


    
    def encode(self, text):
        ids = list(text.encode('utf-8'))

        while True:
            pairs = self.get_pairs(ids)
            mergeable_pairs = {p: self.merges[p] for p in pairs if p in self.merges}


            if not mergeable_pairs:
                break

            pair = min(mergeable_pairs, key=self.merges.get)

            ids = self.merge(ids, pair, self.merges[pair])

        return ids
    
    
    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text


    def get_pairs(self, ids, counts=None):

        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts


    def save_vocab_and_merges(self, path):
        data = {
            'vocab': {},
            'merges': {}
        }
        # Save vocab
        for idx, byte_val in self.vocab.items():
            try:
                data['vocab'][str(idx)] = byte_val.decode('utf-8')
            except UnicodeDecodeError:
                data['vocab'][str(idx)] = byte_val.hex()
        # Save merges
        for (first, second), idx in self.merges.items():
            key = f"{first},{second}"  # Convert tuple to string
            data['merges'][key] = idx
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
            
    def load_vocab(self, path='vocab.json'):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Load vocab
        self.vocab = {}
        for idx_str, value in data['vocab'].items():
            idx = idx_str
            self.vocab[idx] = value.encode('utf-8')
        # Load merges
        self.merges = {}
        for pair_str, idx in data['merges'].items():
            first_str, second_str = pair_str.split(',')
            first, second = int(first_str), int(second_str)
            self.merges[(first, second)] = idx
    
    
    def merge(self, ids, pair, idx):
        id = 0
        newids = []
        while id<len(ids):
            if id < len(ids)-1 and ids[id]==pair[0] and ids[id+1]==pair[1]:
                newids.append(idx)
                id += 2
            else:
                newids.append(ids[id])
                id+=1
        return newids


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--text', type=str, help='Text to train tokenizer on')
    parser.add_argument('-v','--vocab_size', type=int, help='Vocab size for tokenizer')
    parser.add_argument('-o', '--output', default='vocab.json', type=str, help='Output path for vocab and merges')
    parser.add_argument('-p', '--pattern', type=str, help='Regex pattern to split text')
    args = parser.parse_args()
    
    with open(args.text, 'r') as f:
        args.text = f.read()
    
    tokenizer = GPT4Tokenizer(args.output, args.pattern)
    tokenizer.train(args.text, args.vocab_size)
