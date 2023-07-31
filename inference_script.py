# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from configs.fsdp import fsdp_config
from configs.training import train_config

import fire
import torch
import os
import sys
import time
from typing import List
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType
)

from transformers import LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.distributed._shard.checkpoint import FileSystemReader
from inference.safety_utils import get_safety_checker
from inference.model_utils import load_model, load_peft_model
from utils import train_utils
from utils.fsdp_utils import fsdp_auto_wrap_policy


def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=True, # Enable safety check woth Saleforce safety flan t5
    checkpoint_dir: str = None, # [optional] The directory to find checkpoints of the model (no peft)
    **kwargs
):
    # if prompt_file is not None:
    #     assert os.path.exists(
    #         prompt_file
    #     ), f"Provided Prompt file does not exist {prompt_file}"
    #     with open(prompt_file, "r") as f:
    #         user_prompt = "\n".join(f.readlines())
    # elif not sys.stdin.isatty():
    #     user_prompt = "\n".join(sys.stdin.readlines())
    # else:
    #     print("No user prompt provided. Exiting.")
    #     sys.exit(1)

    user_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat does it mean to \"switch the billing setting for creating separate billing documents for each item of the billing due list in the Create Billing Documents app to off\"\n\n###Input:\nby product ID or service confirmation ID). * Group the items (for example, by sold-to party). * Use page variants to save and load custom filter and column layouts that you have created. Filter values that you have set are also saved. You can share saved page variants with other users in the system. Release Items for Billing You can release service order and service confirmation items that you have selected, for billing. After you have released the items, the system creates billing document requests.Note If an item belongs to a service bundle, you must release all items of the bundle simultaneously. Caution \n To ensure that the system consolidates the billing document requests to create only one invoice containing the agreed fixed price, the billing setting for creating separate billing documents for each item of the billing due list in the  Create Billing Documents app must be switched to off.Supported Device Types * Desktop * Tablet \n\n### Response:"


    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
         
            "pad_token": "<PAD>",
        }
    )

    # use FSDP
    if checkpoint_dir is not None:
        train_utils.setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        if torch.distributed.is_initialized():
            torch.cuda.set_device(rank)
            train_utils.setup_environ_flags(rank)
        
        mixed_precision_policy, wrapping_policy = train_utils.get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
   
        model = FSDP(
            model,
            auto_wrap_policy=wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
        )

        model = FSDP(model, device_id=local_rank)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = model.state_dict()
            torch.distributed._shard.checkpoint.load_state_dict(state_dict=state_dict, storage_reader=FileSystemReader(checkpoint_dir))
            model.load_state_dict(state_dict)
            model.to(local_rank)

    
    safety_checker = get_safety_checker(enable_azure_content_safety,
                                        enable_sensitive_topics,
                                        enable_saleforce_content_safety,
                                        )

    # Safety check of the user prompt
    safety_results = [check(user_prompt) for check in safety_checker]
    are_safe = all([r[1] for r in safety_results])
    if are_safe:
        print("User prompt deemed safe.")
        print(f"User prompt:\n{user_prompt}")
    else:
        print("User prompt deemed unsafe.")
        for method, is_safe, report in safety_results:
            if not is_safe:
                print(method)
                print(report)
        print("Skipping the inferece as the prompt is not safe.")
        sys.exit(1)  # Exit the program with an error status

    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    batch = tokenizer(user_prompt, return_tensors="pt")
    batch = {k: v.to("cuda") for k, v in batch.items()}
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            min_length=min_length,
            use_cache=use_cache,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            **kwargs 
        )
    e2e_inference_time = (time.perf_counter()-start)*1000
    print(f"the inference time is {e2e_inference_time} ms")
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Safety check of the model output
    safety_results = [check(output_text) for check in safety_checker]
    are_safe = all([r[1] for r in safety_results])

    if are_safe:
        if rank == 0:
            print("User input and model output deemed safe.")
            print(f"Model output:\n{output_text}")
    else:
        print("Model output deemed unsafe.")
        for method, is_safe, report in safety_results:
            if not is_safe:
                print(method)
                print(report)
                

if __name__ == "__main__":
    fire.Fire(main)
