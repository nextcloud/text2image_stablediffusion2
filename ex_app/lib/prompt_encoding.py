"""Encode SDXL prompts longer than the tokenizer limit by splitting them into chunks."""

import torch


def encode_prompt(pipe, prompt):
    device = pipe._execution_device
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2

    prompt_token_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    prompt_token_ids_2 = tokenizer_2(
        prompt,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    chunk_size = tokenizer.model_max_length - tokenizer.num_special_tokens_to_add(pair=False)
    chunk_size_2 = tokenizer_2.model_max_length - tokenizer_2.num_special_tokens_to_add(pair=False)

    chunk_count = max(
        1,
        (len(prompt_token_ids) + chunk_size - 1) // chunk_size,
        (len(prompt_token_ids_2) + chunk_size_2 - 1) // chunk_size_2,
    )

    prompt_chunks = _chunk_prompt_tokens(tokenizer, prompt_token_ids, chunk_count, chunk_size)
    prompt_chunks_2 = _chunk_prompt_tokens(tokenizer_2, prompt_token_ids_2, chunk_count, chunk_size_2)

    prompt_embeds_list = []
    prompt_embeds_2_list = []
    for chunk_index, (chunk, chunk_2) in enumerate(zip(prompt_chunks, prompt_chunks_2, strict=True)):
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        encoder_output = pipe.text_encoder(input_ids, output_hidden_states=True)
        prompt_embeds_list.append(encoder_output.hidden_states[-2])

        input_ids_2 = torch.tensor([chunk_2], dtype=torch.long, device=device)
        encoder_output_2 = pipe.text_encoder_2(input_ids_2, output_hidden_states=True)
        prompt_embeds_2_list.append(encoder_output_2.hidden_states[-2])
        # sdxl takes its pooled embedding from 2nd text encoder
        if chunk_index == 0:
            pooled_prompt_embeds = encoder_output_2[0]

    dtype = pipe.text_encoder.dtype
    prompt_embeds = torch.cat(prompt_embeds_list, dim=1)
    prompt_embeds_2 = torch.cat(prompt_embeds_2_list, dim=1)
    prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_2), dim=-1).to(device=device, dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)
    return prompt_embeds, pooled_prompt_embeds


def _chunk_prompt_tokens(tokenizer, token_ids, chunk_count, chunk_size):
    prepared_chunks = []
    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_size
        token_chunk = token_ids[start : start + chunk_size]

        prepared_chunks.append(
            tokenizer.prepare_for_model(
                token_chunk,
                add_special_tokens=True,
                max_length=tokenizer.model_max_length,
                padding="max_length",
                return_attention_mask=False,
            )["input_ids"]
        )
    return prepared_chunks
