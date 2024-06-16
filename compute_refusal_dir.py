import torch
import gc
import random

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from tqdm import tqdm

#MODEL_ID = "Mistral-7B-Instruct-v0.2"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Run all prompts in target.txt
ALL_PROMPTS = True

# More samples can help find the direction better.
NUM_PROMPT_SAMPLES = 1024

# Used to skip the first and last layers for the modifications.
SKIP_BEGIN_LAYERS = 1  # Don't mess with the first layer.
SKIP_END_LAYERS = 0

# The layer we will use for the refusal_dir calculation will be floor(LAYER_FRACTION_TO_USE * model.layers).
LAYER_FRACTION_TO_USE = 0.6

# Use a negative scale_factor to "induce" and a positive scale_factor of < 1 to "ablate" less.
SCALE_FACTOR = 1.0

torch.inference_mode()
torch.set_default_device("cpu")
torch.set_grad_enabled(False)

# Load the model on the GPU in quantized type if we can.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    low_cpu_mem_usage=True,
    device_map='auto'
)
model.requires_grad_(False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

layer_idx = int(len(model.model.layers) * LAYER_FRACTION_TO_USE)
print("Layer index for refusal direction: " + str(layer_idx))

with open("selected.txt", "r") as f:
    target = f.readlines()

with open("control.txt", "r") as f:
    control = f.readlines()

if ALL_PROMPTS:
    NUM_PROMPT_SAMPLES = len(target)
    target_instructions = target
    control_instructions = random.sample(control, min(NUM_PROMPT_SAMPLES, len(control)))
else:
    target_instructions = random.sample(target, min(NUM_PROMPT_SAMPLES, len(target)))
    control_instructions = random.sample(control, min(NUM_PROMPT_SAMPLES, len(control)))

target_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True,
                                  return_tensors="pt") for insn in target_instructions]
control_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True,
                                  return_tensors="pt") for insn in control_instructions]

bar_generate = tqdm(total = len(target_instructions) + len(control_instructions), desc = "Generating samples")

# Only return the final hidden state of the layer we care about, and use 'cpu' to save VRAM.
def generate(toks):
    output = model.generate(
        toks.to(model.device),
        use_cache=False,
        max_new_tokens=1,
        return_dict_in_generate=True,
        output_hidden_states=True,
        pad_token_id=tokenizer.eos_token_id
    )
    bar_generate.update(n=1)
    return output.hidden_states[0][layer_idx][:, -1, :].to('cpu') # Final hidden state = -1.

target_hidden = [generate(toks) for toks in target_toks]
control_hidden = [generate(toks) for toks in control_toks]

bar_generate.close()

target_mean = torch.stack(target_hidden).mean(dim=0)
control_mean = torch.stack(control_hidden).mean(dim=0)

refusal_dir = target_mean - control_mean
refusal_dir = refusal_dir.squeeze() / refusal_dir.norm()

torch.save(refusal_dir, MODEL_ID.replace("/", "_") + "_refusal_dir.pt")

# Free memory
del model
gc.collect()
torch.cuda.empty_cache()

# Reload the model in CPU memory with bfloat16 data type
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map='cpu'
)
model.requires_grad_(False)

# Make sure it's on the 'cpu' device.
if refusal_dir.device != model.device:
    refusal_dir = refusal_dir.to(model.device)

# Get the language model component and check it's as expected.
lm_model = model.model
assert hasattr(lm_model, 'layers'), "The model does not have the expected structure."

# Check the ranges are valid.
num_layers = len(lm_model.layers)
assert SKIP_BEGIN_LAYERS >= 0, "SKIP_BEGIN_LAYERS must be >= 0."
assert SKIP_END_LAYERS >= 0, "SKIP_END_LAYERS must be >= 0."
assert SKIP_BEGIN_LAYERS + SKIP_END_LAYERS < num_layers, "SKIP_BEGIN_LAYERS + SKIP_END_LAYERS must be < num_layers."

bar_layers = tqdm(total= (num_layers - (SKIP_BEGIN_LAYERS + SKIP_END_LAYERS)) * 2, desc = "Modifying tensors")

# Cast any ops performed on CPU up to float32... If you have newer CPU might be able to use bfloat16 for this.
# NOTE: Use a negative scale_factor to "induce" and a positive scale_factor of < 1 to "ablate" less.
def modify_tensor(tensor_data, refusal_dir, scale_factor: float = 1.0):
    assert scale_factor <= 1.0, "Using a scale_factor of > 1 doesn't make sense..."
    tensor_float32 = tensor_data.to(torch.float32)
    refusal_dir_float32 = refusal_dir.to(torch.float32)
    tensor_float32 -= scale_factor * torch.matmul(torch.outer(refusal_dir_float32, refusal_dir_float32), tensor_float32)
    tensor_modified = tensor_float32.to(torch.bfloat16)
    bar_layers.update(1)
    return torch.nn.Parameter(tensor_modified)

# Modify the 'self_attn.o_proj.weight' and 'mlp.down_proj.weight' in each chosen layer.
# NOTE: These tensors names are speific to "llama" and may need changing.
#       - See here for others: https://github.com/arcee-ai/mergekit/tree/main/mergekit/_data/architectures
for layer_idx in range(SKIP_BEGIN_LAYERS, num_layers - SKIP_END_LAYERS):
    lm_model.layers[layer_idx].self_attn.o_proj.weight = modify_tensor(
        lm_model.layers[layer_idx].self_attn.o_proj.weight.data, refusal_dir, SCALE_FACTOR
    )
    lm_model.layers[layer_idx].mlp.down_proj.weight = modify_tensor(
        lm_model.layers[layer_idx].mlp.down_proj.weight.data, refusal_dir, SCALE_FACTOR
    )

bar_layers.close()

# Save the modified model and original tokenizer
print("Saving modified model (with original tokenizer)...")
model.save_pretrained("llama3_instruct_fixed")
tokenizer.save_pretrained("llama3_instruct_fixed")