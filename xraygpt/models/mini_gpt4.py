import logging  # Standard logging library for debug/info output
import random   # Used for random selection (e.g., prompt choices)

import torch  # Main PyTorch module for tensor operations and modeling
from torch.cuda.amp import autocast as autocast  # For mixed-precision (automatic casting)
import torch.nn as nn  # Neural network components

from xraygpt.common.registry import registry  # Model registry for registering classes
from xraygpt.models.blip2 import Blip2Base, disabled_train  # BLIP2 base class and helper for freezing training
from xraygpt.models.modeling_llama import LlamaForCausalLM  # Llama model implementation for causal language modeling
from transformers import LlamaTokenizer  # Tokenizer for Llama model
from transformers import StoppingCriteria, StoppingCriteriaList  # For custom stopping in text generation
import csv  # For CSV operations (used in other parts, e.g., rogue eval)
from xraygpt.conversation.conversation import Conversation  # Conversation wrapper for dialogue formatting
from enum import auto, Enum  # For custom Enum classes (e.g., separator styles)
from typing import List, Tuple, Any  # Type hints

# Custom stopping criteria for text generation (used for stopping at specific tokens)
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()  # Call base class constructor
        self.stops = stops  # List of stop token sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Check if the generated sequence ends with any of the stop tokens
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True  # Stop if match found
        return False  # Otherwise, continue

# Enum for separator styles in conversation formatting
class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

# Register the model to a registry under the name "mini_gpt4"
@registry.register_model("mini_gpt4")
class MiniGPT4(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    # Preset config for pretraining
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/xraygpt.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",  # Vision transformer model type
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",  # Q-Former checkpoint path
        img_size=224,  # Input image size
        drop_path_rate=0,  # VIT drop path rate
        use_grad_checkpoint=False,  # Use grad checkpointing for memory
        vit_precision="fp16",  # Precision for VIT
        freeze_vit=True,  # Freeze vision encoder weights
        freeze_qformer=True,  # Freeze Q-Former weights
        num_query_token=32,  # Number of query tokens for Q-Former
        llama_model="",  # Path to Llama model
        prompt_path="",  # Path to prompts file
        prompt_template="",  # Format template for prompts
        max_txt_len=32,  # Maximum text length for generation/training
        low_resource=False,  # Use low-resource mode (8bit, VIT on CPU)
        end_sym='\n',  # End-of-text symbol
    ):
        super().__init__()  # Initialize parent (BLIP2Base)

        self.tokenizer = self.init_tokenizer()  # Initialize tokenizer
        self.low_resource = low_resource  # Save low resource mode flag

        print('Loading VIT')
        # Initialize vision encoder (VIT) and its layer norm
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            # Freeze all parameters in the vision encoder
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()  # Set to eval mode
            self.visual_encoder.train = disabled_train  # Disable training
            # Freeze all parameters in the vision encoder's layer norm
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()  # Set to eval mode
            self.ln_vision.train = disabled_train  # Disable training
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        # Initialize Q-Former and its learnable query tokens
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        # Remove unused heads in Q-Former for efficiency
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        # Load Q-Former weights from pretrained checkpoint
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            # Freeze all Q-Former parameters
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()  # Set to eval mode
            self.Qformer.train = disabled_train  # Disable training
            self.query_tokens.requires_grad = False  # Freeze query tokens
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')

        print('Loading LLAMA')
        # Load Llama tokenizer
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token  # Set pad as EOS for compatibility

        # Load Llama language model, optionally with device_map for low-resource
        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        # Freeze all parameters of Llama model
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        # Linear projection from Q-Former hidden size to Llama hidden size
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = max_txt_len  # Store max text length
        self.end_sym = end_sym  # Store end symbol

        # Load prompt templates (if provided)
        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            # Filter prompts that include the image placeholder
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            # Format prompts with template
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []  # Empty if no prompt path

        print('#'*100)  # Divider for debug output
 
        # print('#'*100)

    # Move VIT and its layer norm to CPU (for low-resource mode)
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    # Encode image to embeddings for downstream text generation
    def encode_img(self, image):
        device = image.device  # Get image tensor device
        if self.low_resource:
            self.vit_to_cpu()  # Move VIT to CPU
            image = image.to("cpu")  # Move image to CPU

        with self.maybe_autocast():  # Use autocast for mixed precision
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)  # Get vision embeddings
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)  # Attention mask

            # Expand query tokens for batch size
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            # Pass through Q-Former
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # Project Q-Former output to Llama input space
            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)  # Attention mask for Llama
        return inputs_llama, atts_llama

    # Wrap image embeddings with prompt text on both sides
    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')  # Split prompt into before/after image part
            # Tokenize both parts
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            # Embed tokens and expand for batch
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            # Concatenate before-prompt, image embeds, after-prompt
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])  # Update attention mask
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img  # If no prompt, return as is

    # Forward method for training
    def forward(self, samples):
        image = samples["image"]  # Get image batch
        img_embeds, atts_img = self.encode_img(image)  # Encode image
        if hasattr(samples, 'question_split'):  # VQA dataset case
            print('VQA Batch')
            vqa_prompt = '###Patient: <Img><ImageHere></Img> '
            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, vqa_prompt)
        elif self.prompt_list:
            prompt = random.choice(self.prompt_list)  # Use random prompt

            # Uncomment for finding-augmented prompts
            # x, y = prompt.split('###Doctor:')
            # x = x + ' '.join(samples['finding']) + y
            # prompt = x + '###Doctor:' + y

            img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)

        self.llama_tokenizer.padding_side = "right"  # Set tokenizer padding side

        text = [t + self.end_sym for t in samples["caption"]]  # Append end symbol

        # Tokenize text captions
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)

        # Replace pad token IDs with -100 for loss masking
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        # Prepare empty targets for image/prompt tokens (not regressed)
        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image.device).fill_(-100)  # +1 for BOS token
        )
        targets = torch.cat([empty_targets, targets], dim=1)  # Concatenate target masks

        batch_size = img_embeds.shape[0]
        # Prepare BOS token embeddings
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]  # Attention mask for BOS

        # Embed the caption tokens
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        # Concatenate all embeddings for Llama input
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,  # Supervised learning target
            )
        loss = outputs.loss  # Training loss

        return {"loss": loss}  # Return loss for optimization
    
    # Inference (test) method for generating text from images (dialogue-based)
    def test(self, samples, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
        
        # Initialize a conversation template
        conv = Conversation(
            system="A chat between a patient and an experienced Doctor."
                "If you are a doctor, please answer the medical questions based on the patient's description. Give the following medical scan: <Img>ImageContent</Img>."
                "You will be able to see the medical scan once I provide it to you. Please answer the patients questions.",
            roles=("Patient", "Doctor"),
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep="###",
            sep2="</s>",
        )

        # Define stop word IDs for generation
        stop_words_ids = [torch.tensor([835]).to(samples["image"].device),
                          torch.tensor([2277, 29937]).to(samples["image"].device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        conv.append_message(conv.roles[1], None)  # Doctor's (empty) message slot

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)  # Encode image
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")  # Add image to conversation

        embs = self.get_context_emb(conv, img_embeds)  # Get full conversation context embeddings

        # Truncate context if exceeding max tokens
        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]

        # Generate text from Llama model
        outputs = self.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        # Remove spurious start or unknown tokens from output
        if output_token[0] == 0:  # <unk>
            output_token = output_token[1:]
        if output_token[0] == 1:  # <s> start
            output_token = output_token[1:]
        # Decode output tokens to text
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # Remove at stop sign
        output_text = output_text.split('Doctor:')[-1].strip()  # Only keep doctor response
        conv.messages[-1][1] = output_text  # Save response in conversation
        return output_text, output_token.cpu().numpy()  # Return text and tokens

    # Build the full context embedding for Llama based on conversation and image
    def get_context_emb(self, conv, img):
        prompt = random.choice(self.prompt_list)  # Choose a prompt template
        text = prompt.split("<Img><ImageHere></Img>")[-1]  # Get post-image prompt text

        # If the last message is an image, append text after it (to maintain context integrity)
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # Last message is image
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], prompt)  # Otherwise, add as a new prompt

        prompt_segs = prompt.split('<ImageHere>')  # Split for text-image-text
        img_list = [img]  # Single image
        # assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # Add BOS token only to the first segment
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]  # Embed text segments
        # Interleave text and image embeddings as per prompt
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)  # Concatenate all for model input
        return mixed_embs

    # Class method to build model from a config dictionary
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        # Instantiate the model with config parameters
        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym
        )

        # Optionally load weights from checkpoint for full model
        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model  # Return constructed model
