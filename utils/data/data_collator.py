import logging

import torch
from transformers.data.data_collator import *


logger = logging.getLogger(__name__)
PAD_TOKEN_LABEL = -100


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_prompt_len: Optional[int] = None
    max_ans_len: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    inference: bool = False

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

         model_inputs = self.decoder_call(batch, return_tensors)

        return model_inputs

    # support decoder-only models
    def decoder_call(self, batch, return_tensors):
        self.tokenizer.padding_side = 'left'
        sources = []
        label_lens = []
        labels = []
        max_len = -1

        for instance in batch:
            label = instance['Instance']['label']
            labels.append(label)
            instruction = self.get_instruction(instance)

            # add bos and eos
            task_input = self.tokenizer.bos_token + instruction
            label = label + self.tokenizer.eos_token

            tokenized_input = self.tokenizer(task_input)["input_ids"]
            tokenized_label = self.tokenizer(label)["input_ids"]

            # (input) for inference, (input + label) for training
            if instance['subset'] in ['dev', 'test']:
                label_lens.append(0)
                if len(tokenized_input) <= limit_input_len:
                    max_len = max(len(tokenized_input), max_len)
                    sources.append(task_input)
                else:
                    max_len = limit_input_len
                    input_wo_label = self.tokenizer.decode(
                        tokenized_input[: limit_input_len],
                        skip_special_tokens=False
                    )
                    sources.append(input_wo_label)
            else:
                if len(tokenized_input) + len(tokenized_label) <= limit_input_len:
                    max_len = max(len(tokenized_input) + len(tokenized_label), max_len)
                    label_lens.append(len(tokenized_label))
                    sources.append(task_input + label)
                else:
                    max_len = self.max_source_length
                    input_w_label = self.tokenizer.decode(
                        (tokenized_input + tokenized_label)[: limit_input_len],
                        skip_special_tokens=False
                    )
                    sources.append(input_w_label)
                    label_lens.append(max(0, limit_input_len - len(tokenized_input)))


        model_inputs = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            padding=self.padding,
            return_tensors=return_tensors,
            truncation=True,
            pad_to_multiple_of=self.pad_to_multiple_of
        )

        label_mask = model_inputs["attention_mask"].bool()
        model_inputs["labels"] = model_inputs['input_ids'].masked_fill(~label_mask, self.label_pad_token_id)

        # loss mask
        max_len = min(max_len, limit_input_len)
        loss_mask = torch.ones((label_mask.shape))
        for k, label_len in enumerate(label_lens):
            loss_mask[k, : max_len - label_len - 1] = 0
        model_inputs['loss_mask'] = loss_mask.masked_fill(~label_mask, 0)

        return model_inputs
