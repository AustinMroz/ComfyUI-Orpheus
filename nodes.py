import json
import os.path

import torch
import torchaudio
import safetensors.torch
from transformers import LlamaForCausalLM, PretrainedConfig
from transformers.generation import logits_process, LogitsProcessorList

from comfy.utils import load_torch_file
from comfy.text_encoders.hunyuan_video import LLAMA3Tokenizer
import folder_paths
from comfy.model_management import get_torch_device

if not "orpheus" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("orpheus", os.path.join(folder_paths.models_dir, "orpheus"))


execution_device = get_torch_device()
tokeniser_length = 128256
TOKENS = {'start_of_text':  128000, 'end_of_text': 128009, 'start_of_speech':
          tokeniser_length + 1, 'end_of_speech': tokeniser_length + 2,
          'start_of_human': tokeniser_length + 3, 'end_of_human':
          tokeniser_length + 4, 'start_of_ai': tokeniser_length + 5,
          'end_of_ai':  tokeniser_length + 6, 'pad_token': tokeniser_length +
          7,}

class ContainsAll(dict):
    def __contains__(self, other):
        return True
    def __getitem__(self, key):
        return super().get(key, (None, {}))

class SnacVAE:
    def __init__(self, state_dict, config):
        from snac import SNAC
        if os.path.isfile(config):
            with open(config, 'r') as f:
                config = json.load(f)
        self.model = SNAC(**config).eval()
        self.model.load_state_dict(state_dict)
    def decode(self, codes):
        with torch.inference_mode():
            try:
                self.model.to(execution_device)
                return self.model.decode(codes)
            finally:
                self.model.to('cpu')
    def encode(self, atensor):
        with torch.inference_mode():
            try:
                self.model.to(execution_device)
                return self.model.encode(atensor)
            finally:
                self.model.to('cpu')

#NOTE: Native VAEEncodeAudio forces incorrect sample rate and can't be used
class OrpheusDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"tokens": ("ORPH_TOKENS",),"vae": ("VAE",),},}
    FUNCTION = "decode"
    RETURN_TYPES = "AUDIO",
    CATEOGRY = "Orpheus"
    def decode(self, tokens, vae):
        #assert len(tokens) % 7 == 0
        tokens = tokens[:len(tokens)//7*7]
        t = torch.tensor(tokens).reshape((-1,7))
        t -= 10 + tokeniser_length
        t -= torch.arange(7, device=t.device) * 4096
        if t.min() < 0 or t.max() > 4096:
            raise ValueError("Invalid codes. Should be impossible. Open an issue.")
        codes = [t[:,0]        .to(execution_device).reshape((1,-1)),
                 t[:,[1,4]]    .to(execution_device).reshape((1,-1)),
                 t[:,[2,3,5,6]].to(execution_device).reshape((1,-1))]
        return {"waveform": vae.decode(codes), "sample_rate": 24000},
class OrpheusEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"vae": ("VAE",), "audio": ("AUDIO",),
                             "formatting": (["BOTH", "PRE", "POST", "NONE"],)},}
    FUNCTION = "encode"
    RETURN_TYPES = ("ORPH_TOKENS",)
    CATEOGRY = "Orpheus"
    def encode(self, vae, audio, formatting):
        w = audio['waveform']
        rate = audio['sample_rate']
        if rate != 24000:
            w = torchaudio.functional.resample(w, rate, 24000)
        #TODO: Better channel processing
        w = w[:,:1]
        w = w.to(execution_device)
        codes = vae.encode(w)
        codes = [codes[0].reshape((-1,1)),
                 codes[1].reshape((-1,2)),
                 codes[2].reshape((-1,4))]
        combined = torch.cat([codes[0][:,[0]],codes[1][:,[0]],codes[2][:,[0,1]],
                             codes[1][:,[1]],codes[2][:,[2,3]]],dim=1)
        combined += 10 + tokeniser_length
        combined += torch.arange(7, device=combined.device) * 4096
        combined = combined.flatten().tolist()
        if formatting == "PRE" or formatting == "BOTH":
            combined = [TOKENS['start_of_ai'], TOKENS['start_of_speech']] + combined
        if formatting == "POST" or formatting == "BOTH":
            combined += [TOKENS['end_of_speech'], TOKENS['end_of_ai']]
        return combined,

class LoadSnacVAE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"snac_model": (folder_paths.get_filename_list('vae'),)},}
    FUNCTION = "loadvae"
    RETURN_TYPES = ("VAE",)
    CATEOGRY = "Orpheus"

    def loadvae(self, snac_model):
        vae_path = folder_paths.get_full_path_or_raise('vae', snac_model)
        config = os.path.join(os.path.split(__file__)[0], 'snac-24khz-config.json')
        state_dict = load_torch_file(vae_path)
        return SnacVAE(state_dict, config),

def convetToSection(section):
    if isinstance(section, str):
        return [TOKENS['start_of_human'], TOKENS['start_of_text']] \
                + tok(section).input_ids \
                + [TOKENS['end_of_text'], TOKENS['end_of_human']]
    elif isinstance(section, dict) and 'waveform' in section:
        w = section['waveform']
        if section['sr'] != 24000:
            w = torchaudio.functional.resample(w, section['sr'], 24000)

class AudioLogitsProcessor(logits_process.LogitsProcessor):
    def __init__(self, start_index):
        self.start_index = start_index
    def __call__(self, input_ids, score):
        offset = (input_ids.size(-1) - self.start_index) % 7
        new_score = torch.zeros_like(score)
        if offset == 0 and input_ids.size(-1) > self.start_index:
            new_score[:,TOKENS['end_of_speech']] = score[:,TOKENS['end_of_speech']]
        code_base = tokeniser_length + 10 + 4096 * offset
        new_score[:,code_base:code_base+4096] = score[:,code_base:code_base+4096]
        return new_score

class LoadOrpheus:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": (folder_paths.get_filename_list('orpheus'),)},}
    FUNCTION = "loadorpheus"
    RETURN_TYPES = ("ORPH_MODEL",)
    CATEOGRY = "Orpheus"

    def loadorpheus(self, model):
        model_path = folder_paths.get_full_path_or_raise('orpheus', model)
        #TODO: Save conf in safetensors?
        conf = os.path.join(os.path.split(__file__)[0], 'orpheus-config.json')
        config = PretrainedConfig.from_json_file(conf)
        sd = safetensors.torch.load_file(model_path)
        #TODO: use this to detect pt/ft and add further tweaks?
        config.vocab_size = sd['lm_head.weight'].size(0)
        model = LlamaForCausalLM.from_pretrained(None, config=config, state_dict=sd)
        return model,

class OrpheusSample:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("ORPH_MODEL",),
                             "prompt": ("ORPH_TOKENS",),
                             "add_start_token": ("BOOLEAN", {"default": True}),
                             },
                "hidden": {"seed": "seed"}}
    FUNCTION = "sample"
    RETURN_TYPES = ("ORPH_TOKENS","ORPH_TOKENS", "ORPH_TOKENS")
    RETURN_NAMES = ("full", "chunk", "generated")
    CATEOGRY = "Orpheus"
    def sample(self, model, prompt, add_start_token, seed=None):
        #Seed is not something reproducible. Just a "reroll" counter
        #TODO: Figure out seed. Likely requires migrating away from transformers
        if add_start_token:
            prompt += [TOKENS['start_of_ai'], TOKENS['start_of_speech']]
            chunk_index = len(prompt)
        else:
            chunk_index = len(prompt) - prompt[::-1].index(TOKENS['start_of_speech'])
        try:
            model.to(execution_device)
            input_ids = torch.tensor(prompt).unsqueeze(0).to(execution_device)
            attention_mask = torch.ones(input_ids.shape, device=model.device)
            start_index = input_ids.size(-1)
            lpl = LogitsProcessorList([AudioLogitsProcessor(start_index)])
            gen_ids = model.generate(logits_processor=lpl,
                                     eos_token_id=TOKENS['end_of_speech'],
                                     max_new_tokens=7*512,
                                     do_sample=True,
                                     temperature=0.6,
                                     top_p=0.95,
                                     repetition_penalty=1.1,
                                     num_return_sequences=1,
                                     input_ids=input_ids,
                                     attention_mask=attention_mask)
        finally:
            model.to('cpu')
        gen_ids = gen_ids.squeeze(0)
        #TODO: Pass tensors so this is view
        return ( gen_ids.tolist() + [TOKENS['end_of_ai']],
                gen_ids[...,chunk_index:-1].tolist(),
                gen_ids[...,start_index:-1].tolist())

class OrpheusPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True})},
                "optional": {"speaker": (['None','tara','leah','jess','leo','dan','mia','zac','zoe'],)}}
    FUNCTION = "encodeprompt"
    RETURN_TYPES = ("ORPH_TOKENS",)
    CATEOGRY = "Orpheus"
    def __init__(self):
        #TODO: properly load this
        tok = LLAMA3Tokenizer()
        self.tokenizer = tok.tokenizer
    def encodeprompt(self, text, speaker='None'):
        if speaker != 'None':
            text = speaker + ": " + text
        #start_of_text is included during tokenization automatically
        tokens = [TOKENS['start_of_human']] \
                + self.tokenizer(text).input_ids \
                + [TOKENS['end_of_text'], TOKENS['end_of_human']]
        return tokens,


class CombinePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"prompt_1": ("ORPH_TOKENS",)},
                "hidden": ContainsAll()}
    FUNCTION = "concat"
    RETURN_TYPES = ("ORPH_TOKENS",)

    #NOTE: most implementation is frontend
    def concat(self, **kwargs):
        ind = 1
        output = []
        while True:
            if ('prompt_%d' % ind) not in kwargs:
                break
            output += kwargs['prompt_%d' % ind]
            ind += 1
        return output,

NODE_CLASS_MAPPINGS = {
  "ORPH_Sample": OrpheusSample,
  "ORPH_Load": LoadOrpheus,
  "ORPH_Prompt": OrpheusPrompt,
  "ORPH_Decode": OrpheusDecode,
  "ORPH_Encode": OrpheusEncode,
  "ORPH_SnacVae": LoadSnacVAE,
  "ORPH_Combine": CombinePrompt,
}
NODE_DISPLAY_NAME_MAPPINGS = {}
