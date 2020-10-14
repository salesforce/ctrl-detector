"""
Copyright (c) 2020, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from multiprocessing import Process
import subprocess
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import json
import fire
import torch
from urllib.parse import urlparse, unquote


model_combine_256: RobertaForSequenceClassification = None
tokenizer: RobertaTokenizer = None
device: str = None

def log(*args):
    print(f"[{os.environ.get('RANK', '')}]", *args, file=sys.stderr)


class RequestHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        query = unquote(urlparse(self.path).query)

        if not query:
            self.begin_content('text/html')

            html = os.path.join(os.path.dirname(__file__), 'index.html')
            self.wfile.write(open(html).read().encode())
            return

        self.begin_content('application/json;charset=UTF-8')

        tokens = tokenizer.encode(query[1:])
        all_tokens = len(tokens) - 2
        tokens = tokens[:tokenizer.max_len - 2]
        used_tokens = len(tokens) - 2
        model = model_combine_256

        tokens = torch.tensor([tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]).unsqueeze(0)
        mask = torch.ones_like(tokens)

        with torch.no_grad():
            logits = model(tokens.to(device), attention_mask=mask.to(device))[0]
            probs = logits.softmax(dim=-1)

        fake, real = probs.detach().cpu().flatten().numpy().tolist()

        self.wfile.write(json.dumps(dict(
            all_tokens=all_tokens,
            used_tokens=used_tokens,
            real_probability=real,
            fake_probability=fake
        )).encode())

    def begin_content(self, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def log_message(self, format, *args):
        log(format % args)


def serve_forever(server, model_combine_256, tokenizer, device):
    log('Process has started; loading the model ...')
    globals()['tokenizer'] = tokenizer
    globals()['device'] = device
    globals()['model_combine_256'] = model_combine_256.to(device)

    log(f'Ready to serve at http://localhost:{server.server_address[1]}')
    server.serve_forever()


def main(port=8080, device='cuda' if torch.cuda.is_available() else 'cpu'):

    data_combine_256 = torch.load('combine_256.pt', map_location='cpu')

    model_name = 'roberta-large'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    model_combine_256 = RobertaForSequenceClassification.from_pretrained('roberta-large')
    model_combine_256.load_state_dict(data_combine_256['model_state_dict'], strict=False)
    model_combine_256.eval()

    print(f'Starting HTTP server on port {port}', file=sys.stderr)
    server = HTTPServer(('0.0.0.0', port), RequestHandler)

    # avoid calling CUDA API before forking; doing so in a subprocess is fine.
    num_workers = int(subprocess.check_output([sys.executable, '-c', 'import torch; print(torch.cuda.device_count())']))

    if num_workers <= 1:
        serve_forever(server, model_combine_256, tokenizer, device)
    else:
        print(f'Launching {num_workers} worker processes...')

        subprocesses = []

        for i in range(num_workers):
            os.environ['RANK'] = f'{i}'
            os.environ['CUDA_VISIBLE_DEVICES'] = f'{i}'
            process = Process(target=serve_forever, args=(server, model_combine_256, tokenizer, device))
            process.start()
            subprocesses.append(process)

        del os.environ['RANK']
        del os.environ['CUDA_VISIBLE_DEVICES']

        for process in subprocesses:
            process.join()


if __name__ == '__main__':
    fire.Fire(main)
