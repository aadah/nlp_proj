import nltk

# input: filename of article to scan, relative to current directory
# output: set of named entity strings
def ner_recognize_file(filename):

    with open(filename, 'r') as f:
        text = f.read()

    text = unicode(text, 'utf-8')
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentences.append(nltk.word_tokenize(sentence))
    tagged_sentences = nltk.pos_tag_sents(tokenized_sentences)
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences)

    entities = set()

    for tree in chunked_sentences:
        for x in tree:
            if type(x) is tuple and "NNP" in x[1]:
                # print x[0]
                entities.add(x[0])
            elif type(x) is nltk.tree.Tree:
                # print string_from_tree(x)
                entities.add(string_from_tree(x))

    return entities

def string_from_tree(tree):
    strings = []
    for item in tree:
        if type(item) is tuple:
            strings.append(item[0])
        elif type(item) is nltk.tree.Tree:
            strings.append(string_from_tree(item))
        else:
            print "hmm"
    return " ".join(strings)


def ner_recognize_files(list_filenames):
    entities = set()
    for filename in list_filenames:
        entities.update(ner_recognize_file(filename))
    return entities

if __name__ == "__main__":
    print "starting"
    #entities = ner_recognize_files(["article.txt", "article2.txt"])
    entities = ner_recognize_file("article.txt")
    print entities, len(entities)
