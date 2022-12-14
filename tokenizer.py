import torch
from transformers import BertTokenizer

def tokenize(sentences, labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    max_len = 0
    for sent in sentences:
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        max_len = max(max_len, len(input_ids))

    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,  
            max_length = max_len,
            pad_to_max_length =True,
            add_special_tokens = True,
            return_attention_mask = True,  
            return_tensors = 'pt', 
            )
        # pad the encoded data
        # encoded_dict.append([0]* (max_len - len(encoded_dict)))

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels

if __name__ == '__main__':
    sentences = ['I love you']
    labels = [0]
    print(tokenize(sentences, labels))