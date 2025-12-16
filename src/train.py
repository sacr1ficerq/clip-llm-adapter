import torch
from tqdm import tqdm

from transformers import get_cosine_schedule_with_warmup


class QwenVisionCaptionTrainer:
    def __init__(self,
                 qwen_model,
                 qwen_tokenizer,
                 vision_encoder,
                 vision_adapter,
                 device,
                 lr=1e-4,
                 weight_decay=1e-3,
                 max_grad_norm=1.0,
                 num_training_steps=None,
                 warmup_ratio=0.05):
        self.qwen_model = qwen_model
        self.tok = qwen_tokenizer
        self.vision_encoder = vision_encoder
        self.adapter = vision_adapter
        self.device = device
        self.max_grad_norm = max_grad_norm

        self.opt = torch.optim.AdamW(self.adapter.parameters(), lr=lr, weight_decay=weight_decay)

        self.sched = None
        if num_training_steps is not None:
            num_warmup_steps = int(warmup_ratio * num_training_steps)
            self.sched = get_cosine_schedule_with_warmup(
                self.opt, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
            )

        self.pad_id = self.tok.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tok.eos_token_id

    def train_one_epoch(self, train_loader) -> float:
        self.adapter.train()
        total = 0.0

        for batch in tqdm(train_loader, desc="Training"):
            self.opt.zero_grad(set_to_none=True)
            loss = self._forward_batch(batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), self.max_grad_norm)

            self.opt.step()
            if self.sched is not None:
                self.sched.step()

            total += loss.item()

        return total / max(1, len(train_loader))

    def _forward_batch(self, batch):
        pixel_values = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            vis_tokens = self.vision_encoder(pixel_values=pixel_values).last_hidden_state  # (B,S,Dv)

        vis_embeds = self.adapter(vis_tokens)  # (B,T,Dq)
        txt_embeds = self.qwen_model.get_input_embeddings()(input_ids)  # (B,L,Dq)

        inputs_embeds = torch.cat([vis_embeds, txt_embeds], dim=1)  # (B,T+L,Dq)
        vis_attn = torch.ones(input_ids.size(0), vis_embeds.size(1), device=self.device, dtype=attention_mask.dtype)
        full_attn = torch.cat([vis_attn, attention_mask], dim=1)  # (B,T+L)

        labels = input_ids.clone()
        labels[input_ids == self.pad_id] = -100
        labels = torch.cat([torch.full((labels.size(0), vis_embeds.size(1)), -100, device=self.device, dtype=labels.dtype),
                            labels], dim=1)  # (B,T+L)

        out = self.qwen_model(inputs_embeds=inputs_embeds, attention_mask=full_attn, labels=labels)
        return out.loss

    @torch.no_grad()
    def validate(self, val_loader):
        self.adapter.eval()
        total = 0.0

        for batch in tqdm(val_loader, desc="Validating"):
            loss = self._forward_batch(batch)
            total += loss.item()

        return total / max(1, len(val_loader))

    @torch.no_grad()
    def generate(self, pixel_values, max_new_tokens=40, num_beams=3):
        self.adapter.eval()

        vis_tokens = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        vis_embeds = self.adapter(vis_tokens)

        bsz = pixel_values.size(0)
        attn = torch.ones(bsz, vis_embeds.size(1), device=self.device, dtype=torch.long)

        gen_ids = self.qwen_model.generate(
            inputs_embeds=vis_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.pad_id,
        )

        return self.tok.batch_decode(gen_ids, skip_special_tokens=True)
