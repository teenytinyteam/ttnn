import re


class DataLoader:

    def __init__(self, filename):
        self.filename = filename

        with open(self.filename, 'r', encoding='utf-8') as f:
            self.text = f.read().lower()

        self.vocabulary = sorted(set(self.split_text(self.text)))
        self.word2index = {word: index for index, word in enumerate(self.vocabulary)}
        self.index2word = {index: word for index, word in enumerate(self.vocabulary)}

    @staticmethod
    def split_text(text):
        words = re.split(r'([,.:;?_!"()\']|\s)', text.lower())
        return [t.strip() for t in words if t.strip()]

    def encode(self, text):
        words = self.split_text(text)
        return [self.word2index[word] for word in words]

    def decode(self, tokens):
        text = " ".join([self.index2word[index] for index in tokens])
        return re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)


dataset = DataLoader('../one-day.txt')

print('Total number of character: ', len(dataset.text))
print('Word to Index: ', dataset.word2index)

sentence = '"Be careful," his mother says.'
ids = dataset.encode(sentence)

print('Text: ', sentence)
print('Encode: ', ids)
print('Decode: ', dataset.decode(ids))
