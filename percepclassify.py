import sys
import os
import re

all_txt_files = []


def read_data():
    test_path = sys.argv[2]
    for root_path, dirs, files in os.walk(test_path):
        for f in files:
            if f.endswith(".txt") and f != "README.txt":
                all_txt_files.append(os.path.join(root_path, f))


def prediction():
    positive_negative_bias = 0
    positive_negative_weight_vector = {}
    truthful_deceptive_bias = 0
    truthful_deceptive_weight_vector = {}

    model_path = sys.argv[1]
    print(model_path)
    f = open(model_path, "r")
    positive_negative = False
    truthful_deceptive = False
    for line in f:
        if line.strip() == "POSITIVE_NEGATIVE":
            positive_negative = True
            truthful_deceptive = False
        elif line.strip() == "TRUTHFUL_DECEPTIVE":
            positive_negative = False
            truthful_deceptive = True
        else:
            detail = line.split(":")
            key = detail[0].strip()
            value = detail[1].strip()
            if positive_negative:
                if key == "BIAS":
                    positive_negative_bias = float(value)
                else:
                    positive_negative_weight_vector[key] = float(value)
            elif truthful_deceptive:
                if key == "BIAS":
                    truthful_deceptive_bias = float(value)
                else:
                    truthful_deceptive_weight_vector[key] = float(value)
    f.close()

    output_f = open("percepoutput.txt", "w")
    for doc_path in all_txt_files:
        pn_word_count = {}
        pn_instance_vector = {}
        td_word_count = {}
        td_instance_vector = {}
        f = open(doc_path, "r")
        content = f.read()
        clean_content = re.sub(r'[^\w\s]', '', content).lower()
        word_arr = clean_content.split()
        for word in word_arr:
            if word in positive_negative_weight_vector:
                pn_word_count[word] = pn_word_count.get(word, 0) + 1
                pn_instance_vector[word] = 1
            if word in truthful_deceptive_weight_vector:
                td_word_count[word] = td_word_count.get(word, 0) + 1
                td_instance_vector[word] = 1
        f.close()

        positive_negative_total_sum = positive_negative_bias
        truthful_deceptive_total_sum = truthful_deceptive_bias
        for key, value in pn_instance_vector.items():
            if value == 1:
                positive_negative_total_sum += positive_negative_weight_vector[key]
        for key, value in td_instance_vector.items():
            if value == 1:
                truthful_deceptive_total_sum += truthful_deceptive_weight_vector[key]

        curr_prediction = ""
        if truthful_deceptive_total_sum > 0:
            curr_prediction += "truthful "
        else:
            curr_prediction += "deceptive "
        if positive_negative_total_sum > 0:
            curr_prediction += "positive "
        else:
            curr_prediction += "negative "
        curr_prediction += doc_path + "\n"
        output_f.write(curr_prediction)

    output_f.close()


if __name__ == "__main__":
    read_data()
    prediction()
