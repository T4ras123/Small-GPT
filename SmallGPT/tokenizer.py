import regex as re
import argparse 
import json

class GPT4Tokenizer():
    def __init__(self):
        self.vocab = {idx : bytes([idx]) for idx in range(256)}
        self.merges = dict()
 
    
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
          
                
    def train(self, text, vocab_size):

        assert vocab_size >= 256

        num_merges = vocab_size - 256

        text_splitted = text

        ids = list(text_splitted.encode("utf-8"))

        for i in range(num_merges):
            stats = self.get_pairs(ids) 
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.save_vocab_and_merges()

    
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
            

    def get_pairs(self, ids, counts=None):

        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return counts
    

      
    def save_vocab_and_merges(self, path='./src/vocab.json'):
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
    parser.add_argument('--text', type=str, help='Text to train tokenizer on')
    parser.add_argument('--vocab_size', type=int, help='Vocab size for tokenizer')
    args = parser.parse_args()
    
    with open(args.text, 'r') as f:
        args.text = f.read()
    
    tokenizer = GPT4Tokenizer()
    tokenizer.train(args.text, args.vocab_size)
