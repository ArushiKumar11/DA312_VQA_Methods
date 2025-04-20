import torch
import numpy as np
import collections
from tqdm import tqdm
from torch.nn import functional as nnf
from transformers import GPT2Tokenizer
from torch.cuda.amp import autocast

def treebank_tokenize(s):
    from nltk.tokenize import TreebankWordTokenizer
    return TreebankWordTokenizer().tokenize(s)

def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    generated=None,
    entry_length=65,
    temperature=1.0,
    stop_token: str = "<|endoftext|>",
):
    """Generate text using beam search"""
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    
    with torch.no_grad():
        for i in range(entry_length):
            outputs = model.gpt2_model(inputs_embeds=generated)
            logits = outputs.logits

            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            next_token_embed = model.gpt2_model.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            
            if is_stopped.all():
                break
                
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    
    return output_texts

def compute_f1(gold_toks, pred_toks):
    """Compute F1 score between prediction and ground truth"""
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
        
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def evaluate_model(model, dataloader, device):
    """Evaluate the model on the given dataloader"""
    model.eval()
    model = model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    bleu_avg1 = 0.
    f1_avg = 0.
    acc = 0.
    acc_yn = 0.
    c_yn = 1e-9
    
    results = []
    
    for batch in tqdm(dataloader, desc="Testing"):
        images, tokens, mask, q_len = batch['image'], batch['tokens'], batch['mask'], batch['q_len']
        images = images.to(device)
        tokens = tokens.to(device)
        mask = mask.to(device)
        q_len = q_len.to(device)

        with autocast(dtype=torch.float16):
            with torch.no_grad():
                embed = model.generate(images, tokens, mask, q_len)
                out_text = generate_beam(model, tokenizer, generated=embed, entry_length=30, temperature=1)[0]

        out_text = out_text.split("<|endoftext|>")[0]

        question = batch['questions'][0]
        answer = batch['answers'][0]
        
        result = {
            'question': question,
            'true_answer': answer,
            'predicted_answer': out_text
        }
        results.append(result)

        # Calculate metrics
        if out_text.lower() == answer.lower():
            acc += 1
            
        if answer.lower() == 'yes' or answer.lower() == 'no':
            if out_text.lower() == answer.lower():
                acc_yn += 1
            c_yn += 1

        from nltk.translate.bleu_score import sentence_bleu
        reference = [str(answer.lower())]
        candidate = [out_text]
        
        try:
            bleu_1 = sentence_bleu(reference[0], candidate[0], weights=(1, 0, 0, 0))
        except:
            bleu_1 = 0
            
        f1_avg += compute_f1(tokenizer.encode(reference[0]), tokenizer.encode(candidate[0]))
        bleu_avg1 += bleu_1

    # Calculate overall metrics
    metrics = {
        "BLEU": round(bleu_avg1 / len(dataloader), 3),
        "F1": round(f1_avg / len(dataloader), 3),
        "Accuracy": round(acc / len(dataloader), 3),
        "Accuracy YN": round(acc_yn / c_yn, 3) if c_yn > 1e-9 else 0
    }
    
    return results, metrics