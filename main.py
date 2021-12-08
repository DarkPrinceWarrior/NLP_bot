import nltk_utils


if __name__ == "__main__":
    sentence = "How long are you doing it?"
    new_sentence = nltk_utils.tokenize(sentence)
    new_sentence2 = list()
    for element in new_sentence:
        new_sentence2.append(nltk_utils.stemming(element))
    new_sentence = new_sentence2
    print(new_sentence)