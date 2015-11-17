import nltk


def ner_recognize_file(filename):

    with open(filename, 'r') as f:
        text = f.read()

    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentences.append(nltk.word_tokenize(sentence))
    tagged_sentences = nltk.pos_tag_sents(tokenized_sentences)
    chunked_sentences = nltk.ne_chunk_sents(tagged_sentences)


    print chunked_sentences


def ner_recognize_files(list_filenames):
    for filename in list_filenames:
        ner_recognize_file(filename)
