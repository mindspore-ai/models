from model_utils.config import config
from eval import run_gru_eval
import src.tokenization

#eval
print("Start run eval,this may take 5 mins......")
run_gru_eval()
print("run_gru_eval is successful")

# get translation
eval_file = open(config.output_file, mode='r', encoding='utf-8', buffering=True)
original_file = open("./data/test.en", mode='r', encoding='utf-8',
                     buffering=True)

def translate():
    tokenizer = src.tokenization.WhiteSpaceTokenizer(vocab_file="./data/vocab.en")
    num = 1
    for line in eval_file:
        print("\nthe " + str(num) + "th sentence")
        contents = original_file.readline()
        contents = contents.strip('\n')
        print("original:", contents)
        token_ids = [int(x) for x in line.strip().split()]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        sent = " ".join(tokens)
        sent = sent.split("<eos>")[0]
        print("Translations:", sent.strip())
        #print("\n")
        num = num + 1
        if num > 50:
            break
translate()
