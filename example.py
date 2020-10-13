"""
Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_combine_256 = torch.load('combine_256.pt', map_location='cpu')

    model_name = 'roberta-large'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    model_combine_256 = RobertaForSequenceClassification.from_pretrained('roberta-large')
    model_combine_256.load_state_dict(data_combine_256['model_state_dict'], strict=False)
    model_combine_256.eval()

    # Enter input text below
    text = ""
    if not text:
        raise ValueError('Input text string is empty')
    tokens = tokenizer.encode(text)
    all_tokens = len(tokens)
    tokens = tokens[:tokenizer.max_len - 2]
    used_tokens = len(tokens)
    if used_tokens <= 64:
        print('Warning: model performance degrades for shorter inputs.')
    model = model_combine_256

    tokens = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0)
    mask = torch.ones_like(tokens)

    with torch.no_grad():
        logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
        probs = logits.softmax(dim=-1)

    fake, real = probs.detach().cpu().flatten().numpy().tolist()
    print("Prob fake:", fake, "Prob real:", real)
