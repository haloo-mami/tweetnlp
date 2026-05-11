#reading the file 
def load_data(text_file, label_file):
    with open(text_file, "r") as f:
        texts=f.readlines()
    with open(label_file, "r") as f:
        labels=f.readlines()
    
    texts=[t.strip() for t in texts]
    labels=[int(l.strip()) for l in labels]

    return texts, labels

#splitting the tweets into words
def tokenize(text):
    return text.lower().split()

#building a vocabulary i.e. mapping words to numbers
def build_vocab(texts):
    vocab={}
    index=1
    for t in texts:
        words=tokenize(t)

        for word in words:
            if word not in vocab:
                vocab[word]=index
                index+=1
    return vocab

#converting tweet to numbers
def encode_text(text,vocab):
    words=tokenize(text)
    return [vocab.get(word,0) for word in words]