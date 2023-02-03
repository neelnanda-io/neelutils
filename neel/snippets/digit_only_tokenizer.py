# %%
from neel.imports import *

d_vocab = 50278
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
eos_token = "<|EOS|>"
bos_token = "<|BOS|>"
pad_token = "<|PAD|>"
tokenizer.add_special_tokens({"eos_token": eos_token, "bos_token": bos_token})
print(tokenizer)
tokenizer._tokenizer.save("toks.json")
js = json.load(open("toks.json"))
offset = len(js["model"]["vocab"]) - len(js["model"]["merges"])

vocab_copy = js["model"]["vocab"]
merges_copy = js["model"]["merges"]
new_vocab = {}
new_vocab[eos_token] = 0
new_vocab[bos_token] = 1
new_vocab[pad_token] = 2
# offset += 3 - 2 # Adjust for new special tokens
index = 3
new_merges_list = []

offset
for k, v in vocab_copy.items():
    if v < 2:
        continue
    elif re.match("Ġ\d\d.*", k) or re.match("\d\d.*", k):
        # print("Removing key", k)
        continue
    else:
        if re.match(".*\d.*", k):
            print("Not removing digit containing key", k)
        new_vocab[k] = index
        index += 1
        if v >= offset:
            new_merges_list.append(merges_copy[v - offset])
            # print(merges_copy[v-offset], k)
# %%
import copy

new_js = copy.deepcopy(js)
new_js["added_tokens"] = new_js["added_tokens"][2:-2]
for special_token in new_js["added_tokens"]:
    print(index, special_token["content"] in new_vocab)
    print(special_token["content"])
    new_vocab[special_token["content"]] = index
    index += 1

new_js["model"]["vocab"] = new_vocab
new_js["model"]["merges"] = new_merges_list
open("maths_tokens.json", "w").write(json.dumps(new_js))

# %%
from transformers import PreTrainedTokenizerFast

new_tokenizer = PreTrainedTokenizerFast(tokenizer_file="maths_tokens.json")
print(new_tokenizer)
new_tokenizer.add_special_tokens(
    {"eos_token": eos_token, "bos_token": bos_token, "pad_token": pad_token}
)
new_tokenizer.push_to_hub("NeelNanda/gpt-neox-tokenizer-digits")
print(new_tokenizer)

new_tokenizer_2 = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits")

# %%
print(new_tokenizer_2.encode("123+ 32563872+ 39872398743+39 84 2y39"))
print(
    new_tokenizer_2.batch_decode(
        new_tokenizer_2.encode("123+ 32563872+ 39872398743+39 84 2y39"),
        clean_up_tokenization_spaces=False,
    )
)
print(new_tokenizer_2.encode("Hello World"))
# %%
