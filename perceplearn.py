import sys
import os
import re
import random

all_txt_files = []
txt_details = {}
# stopwords = set()
tokens = set()


def read_data():
    train_path = sys.argv[1]
    for root_path, dirs, files in os.walk(train_path):
        if (re.search('positive', root_path.lower()) or re.search('negative', root_path.lower())) \
                and (re.search('truthful', root_path.lower()) or re.search('deceptive', root_path.lower())):
            for f in files:
                if f.endswith(".txt") and f != "README.txt":
                    all_txt_files.append(os.path.join(root_path, f))


# def create_stopwords():
#     f = open("stopwords.txt")
#     for line in f:
#         stopwords.add(line.strip())
#     f.close()


def tokenization():
    for curr in all_txt_files:
        f = open(curr, "r")
        content = f.read()
        clean_content = re.sub(r'[^\w\s]', '', content).lower()
        word_arr = clean_content.split()
        temp_arr = []
        for word in word_arr:
            temp_arr.append(word)
            # if word not in stopwords:
            tokens.add(word)
        f.close()
        txt_details[curr] = temp_arr


def create_vanilla():
    """Positive/Negative Classifier"""
    print("Vanilla ----- Positive/Negative")
    max_iteration = 10
    total_guess = 0
    correct_guess = 0
    filler = [0] * len(tokens)
    weight_vector = dict(zip(tokens, filler))
    bias = 0
    for i in range(max_iteration):
        print("=============== ITER", i, "===============")
        all_doc = all_txt_files
        random.shuffle(all_doc)

        for doc_path in all_doc:
            # curr_word_count = dict(zip(tokens, filler))
            curr_instance_vector = dict(zip(tokens, filler))

            # f = open(doc_path, "r")
            # content = f.read()
            # clean_content = re.sub(r'[^\w\s]', '', content).lower()
            # word_arr = clean_content.split()
            word_arr = txt_details.get(doc_path)
            for word in word_arr:
                # if word not in stopwords:
                #     curr_word_count[word] = curr_word_count.get(word) + 1
                    curr_instance_vector[word] = 1
            # f.close()

            total_sum = bias
            for key, value in curr_instance_vector.items():
                if value == 1:
                    total_sum += weight_vector[key]

            if re.search('positive', doc_path.lower()) and total_sum > 0 or \
                    re.search('negative', doc_path.lower()) and total_sum < 0:
                correct_guess += 1
                total_guess += 1
                continue
            else:
                total_guess += 1
                if re.search('positive', doc_path.lower()):
                    for key, value in curr_instance_vector.items():
                        if value == 1:
                            weight_vector[key] = weight_vector.get(key) + 1
                    bias += 1
                elif re.search('negative', doc_path.lower()):
                    for key, value in curr_instance_vector.items():
                        if value == 1:
                            weight_vector[key] = weight_vector.get(key) - 1
                    bias -= 1
        print(correct_guess, "/", total_guess, "--", correct_guess/total_guess, "%")
    output_f = open("vanillamodel.txt", "w")
    output_f.write("POSITIVE_NEGATIVE\n")
    output_f.write("BIAS:" + str(bias) + "\n")
    for key, value in weight_vector.items():
        output_f.write(key + ":" + str(value) + "\n")

    """Truthful/Deceptive Classifier"""
    print("\nVanilla ----- Truthful/Deceptive")
    max_iteration = 10
    total_guess = 0
    correct_guess = 0
    filler = [0] * len(tokens)
    weight_vector = dict(zip(tokens, filler))
    bias = 0
    for i in range(max_iteration):
        print("=============== ITER", i, "===============")
        all_doc = all_txt_files
        random.shuffle(all_doc)

        for doc_path in all_doc:
            # curr_word_count = dict(zip(tokens, filler))
            curr_instance_vector = dict(zip(tokens, filler))

            # f = open(doc_path, "r")
            # content = f.read()
            # clean_content = re.sub(r'[^\w\s]', '', content).lower()
            # word_arr = clean_content.split()
            word_arr = txt_details.get(doc_path)
            for word in word_arr:
                # if word not in stopwords:
                #     curr_word_count[word] = curr_word_count.get(word) + 1
                    curr_instance_vector[word] = 1
            # f.close()

            total_sum = bias
            for key, value in curr_instance_vector.items():
                if value == 1:
                    total_sum += weight_vector[key]

            if re.search('truthful', doc_path.lower()) and total_sum > 0 or \
                    re.search('deceptive', doc_path.lower()) and total_sum < 0:
                correct_guess += 1
                total_guess += 1
                continue
            else:
                total_guess += 1
                if re.search('truthful', doc_path.lower()):
                    for key, value in curr_instance_vector.items():
                        if value == 1:
                            weight_vector[key] = weight_vector.get(key) + 1
                    bias += 1
                elif re.search('deceptive', doc_path.lower()):
                    for key, value in curr_instance_vector.items():
                        if value == 1:
                            weight_vector[key] = weight_vector.get(key) - 1
                    bias -= 1

        print(correct_guess, "/", total_guess, "--", correct_guess / total_guess, "%")
    output_f.write("TRUTHFUL_DECEPTIVE\n")
    output_f.write("BIAS:" + str(bias) + "\n")
    for key, value in weight_vector.items():
        output_f.write(key + ":" + str(value) + "\n")


def create_averaged():
    """Positive/Negative Classifier"""
    print("\nAveraged ----- Positive/Negative")
    max_iteration = 10
    total_guess = 0
    correct_guess = 0
    filler = [0] * len(tokens)
    weight_vector = dict(zip(tokens, filler))
    cached_weight_vector = dict(zip(tokens, filler))
    bias = 0
    cached_bias = 0
    counter = 1

    for i in range(max_iteration):
        print("=============== ITER", i, "===============")
        all_doc = all_txt_files
        random.shuffle(all_doc)

        for doc_path in all_doc:
            # curr_word_count = dict(zip(tokens, filler))
            curr_instance_vector = dict(zip(tokens, filler))

            # f = open(doc_path, "r")
            # content = f.read()
            # clean_content = re.sub(r'[^\w\s]', '', content).lower()
            # word_arr = clean_content.split()
            word_arr = txt_details.get(doc_path)
            for word in word_arr:
                # if word not in stopwords:
                #     curr_word_count[word] = curr_word_count.get(word) + 1
                    curr_instance_vector[word] = 1
            # f.close()

            total_sum = bias
            for key, value in curr_instance_vector.items():
                if value == 1:
                    total_sum += weight_vector[key]

            if re.search('positive', doc_path.lower()) and total_sum > 0 or \
                    re.search('negative', doc_path.lower()) and total_sum < 0:
                correct_guess += 1
                total_guess += 1
                continue
            else:
                total_guess += 1
                if re.search('positive', doc_path.lower()):
                    for key, value in curr_instance_vector.items():
                        if value == 1:
                            weight_vector[key] = weight_vector.get(key) + 1
                            cached_weight_vector[key] = cached_weight_vector.get(key) + counter
                    bias += 1
                    cached_bias += counter
                elif re.search('negative', doc_path.lower()):
                    for key, value in curr_instance_vector.items():
                        if value == 1:
                            weight_vector[key] = weight_vector.get(key) - 1
                            cached_weight_vector[key] = cached_weight_vector.get(key) - counter
                    bias -= 1
                    cached_bias -= counter
            counter += 1
        print(correct_guess, "/", total_guess, "--", correct_guess / total_guess, "%")
    output_f = open("averagedmodel.txt", "w")
    output_f.write("POSITIVE_NEGATIVE\n")
    averaged_bias = bias - (cached_bias/counter)
    output_f.write("BIAS:" + str(averaged_bias) + "\n")
    for key, value in weight_vector.items():
        curr_averaged_weight = value - (cached_weight_vector[key]/counter)
        output_f.write(key + ":" + str(curr_averaged_weight) + "\n")

    """Truthful/Deceptive Classifier"""
    print("\nAveraged ----- Truthful/Deceptive")
    max_iteration = 10
    total_guess = 0
    correct_guess = 0
    filler = [0] * len(tokens)
    weight_vector = dict(zip(tokens, filler))
    cached_weight_vector = dict(zip(tokens, filler))
    bias = 0
    cached_bias = 0
    counter = 1

    for i in range(max_iteration):
        print("=============== ITER", i, "===============")
        all_doc = all_txt_files
        random.shuffle(all_doc)

        for doc_path in all_doc:
            # curr_word_count = dict(zip(tokens, filler))
            curr_instance_vector = dict(zip(tokens, filler))

            # f = open(doc_path, "r")
            # content = f.read()
            # clean_content = re.sub(r'[^\w\s]', '', content).lower()
            # word_arr = clean_content.split()
            word_arr = txt_details.get(doc_path)
            for word in word_arr:
                # if word not in stopwords:
                #     curr_word_count[word] = curr_word_count.get(word) + 1
                    curr_instance_vector[word] = 1
            # f.close()

            total_sum = bias
            for key, value in curr_instance_vector.items():
                if value == 1:
                    total_sum += weight_vector[key]

            if re.search('truthful', doc_path.lower()) and total_sum > 0 or \
                    re.search('deceptive', doc_path.lower()) and total_sum < 0:
                correct_guess += 1
                total_guess += 1
                continue
            else:
                total_guess += 1
                if re.search('truthful', doc_path.lower()):
                    for key, value in curr_instance_vector.items():
                        if value == 1:
                            weight_vector[key] = weight_vector.get(key) + 1
                            cached_weight_vector[key] = cached_weight_vector.get(key) + counter
                    bias += 1
                    cached_bias += counter
                elif re.search('deceptive', doc_path.lower()):
                    for key, value in curr_instance_vector.items():
                        if value == 1:
                            weight_vector[key] = weight_vector.get(key) - 1
                            cached_weight_vector[key] = cached_weight_vector.get(key) - counter
                    bias -= 1
                    cached_bias -= counter
            counter += 1

        print(correct_guess, "/", total_guess, "--", correct_guess / total_guess, "%")
    output_f.write("TRUTHFUL_DECEPTIVE\n")
    averaged_bias = bias - (cached_bias / counter)
    output_f.write("BIAS:" + str(averaged_bias) + "\n")
    for key, value in weight_vector.items():
        curr_averaged_weight = value - (cached_weight_vector[key] / counter)
        output_f.write(key + ":" + str(curr_averaged_weight) + "\n")


if __name__ == "__main__":
    read_data()
    # create_stopwords()
    tokenization()
    create_vanilla()
    create_averaged()
