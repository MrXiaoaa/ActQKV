import torch
import math
import json
import os
from datetime import datetime
class GreedySearch:
    def __init__(self, model, tokenizer):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.past_kv = None

    def clear(self):
        self.past_kv = None

    def _process_texts(self, input_text):
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text)

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0).cuda()

        return model_inputs


    def generate(self, text=None, input_ids=None, **kwargs):
        if input_ids is None:
            model_inputs = self._process_texts(text)
            input_ids = model_inputs['input_ids']

        with torch.inference_mode():
            result = self._decode(input_ids, **kwargs)
        return result



    def _decode(self, input_ids, max_length=100, extra_end_token_ids=[], chunk_size: int = 4096, output=False, **kwargs):
        if input_ids.dim() == 1:
            input_ids = input_ids[None, :]
        input_ids = input_ids.cuda()
        attention_mask = torch.ones_like(input_ids)
        mask = input_ids.eq(self.tokenizer.pad_token_id)
        attention_mask[mask] = 0
        assert input_ids.size(0) == 1
        length = input_ids.size(1)
        end_token_ids = extra_end_token_ids + [self.tokenizer.eos_token_id]
        logits = None
        past_key_values = self.past_kv
        if output:
            output_text = ""

        log_probs = []
        num_tokens = 0

        for i in range(max_length + 1):
            if i == 0:
                if chunk_size is None:
                    chunk_size = input_ids.size(1)
                for st in range(0, input_ids.size(1) - 1, chunk_size):
                    ed = min(input_ids.size(1) - 1, st + chunk_size)
                    out = self.model(
                        input_ids=input_ids[:, st:ed],
                        attention_mask=attention_mask[:, :ed],
                        use_cache=True,
                        return_dict=True,
                        past_key_values=past_key_values,
                        **kwargs
                    )
                    logits, past_key_values = out.logits, out.past_key_values
                
                out = self.model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values,
                    **kwargs
                )
                logits, past_key_values = out.logits, out.past_key_values

            else:
                out = self.model(
                    input_ids=input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    **kwargs
                )
                logits, past_key_values = out.logits, out.past_key_values

            log_softmax = torch.log_softmax(logits, dim=-1)
            print("log_softmax",log_softmax)
            word = logits.argmax(dim=-1)
            print("word",word)
            word_log_prob = log_softmax.gather(2, word.unsqueeze(-1)).squeeze(-1)
            print("word_log_prob",word_log_prob)
            print("type:word_log_prob",type(word_log_prob))
            log_probs.append(word_log_prob.item())
            num_tokens += 1

            if word.item() in end_token_ids or i == max_length:
                break

            input_ids = torch.cat((input_ids, word.view(1, 1)), dim=-1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((attention_mask.size(0), 1), dtype=torch.int, device=attention_mask.device)),
                dim=-1
            )
            if output:
                tmp = self.tokenizer.decode(input_ids.squeeze(0)[length:])
                if len(tmp) > len(output_text):
                    import sys               
                    sys.stdout.write(tmp[len(output_text):])
                    sys.stdout.flush()
                    output_text = tmp

        self.past_kv = past_key_values

        perplexity = None
        if num_tokens > 0:
            avg_neg_log_prob = -sum(log_probs) / num_tokens
            perplexity = math.exp(avg_neg_log_prob)

        decode_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "output_tokens": self.tokenizer.decode(input_ids.squeeze(0)[length:]),
            "num_tokens": num_tokens,
            "perplexity": perplexity
        }

        json_file_path = './result/ppl/baseline_inf_decode_results-topk46.json'  # 替换为您想要保存的实际路径
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

        # try:
        #     with open(json_file_path, 'r', encoding='utf-8') as f:
        #         data = json.load(f)
        # except (FileNotFoundError, json.JSONDecodeError):
        #     data = []
        data = []
        data.append(decode_result)

        with open(json_file_path, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        if output:
            sys.stdout.write("\n")
            sys.stdout.flush()

        return [decode_result["output_tokens"]]