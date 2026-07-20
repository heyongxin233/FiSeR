import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel

from networks.base_model import BaseModel
from utils.util import resolve_hf_pretrained_path


class CLIPModelLora(nn.Module):
    def __init__(
        self,
        name="openai/clip-vit-large-patch14",
        num_classes=1,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
    ):
        super().__init__()
        self.pretrained_name = resolve_hf_pretrained_path(name)
        self.model = CLIPModel.from_pretrained(self.pretrained_name)
        self.vision_tower = self.model.vision_model
        self.vision_tower.requires_grad_(False)
        self.model.text_model.requires_grad_(False)
        self.model.visual_projection.requires_grad_(False)
        self.model.text_projection.requires_grad_(False)
        self.model.logit_scale.requires_grad_(False)

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.vision_tower_lora = get_peft_model(self.vision_tower, lora_config)
        self.model.vision_model = self.vision_tower_lora

        projection_dim = int(self.model.config.projection_dim)
        self.model.fc = nn.Linear(projection_dim, num_classes)
        torch.nn.init.normal_(self.model.fc.weight.data, 0.0, 0.02)
        torch.nn.init.zeros_(self.model.fc.bias.data)

    def encode_text(self, input_ids, attention_mask):
        text_outputs = self.model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_hidden_states,
            return_dict=self.model.config.use_return_dict,
        )
        text_embeds = text_outputs[1]
        return self.model.text_projection(text_embeds)

    def encode_image(self, img):
        vision_outputs = self.vision_tower_lora(
            pixel_values=img,
            output_attentions=self.model.config.output_attentions,
            output_hidden_states=self.model.config.output_hidden_states,
            return_dict=self.model.config.use_return_dict,
        )
        pooled_output = vision_outputs[1]
        return self.model.visual_projection(pooled_output)

    def forward(self, img, input_ids=None, attention_mask=None, cla=False):
        image_embeds = self.encode_image(img)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        classhead = self.model.fc(image_embeds).squeeze(1)
        if cla:
            return classhead

        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask are required when cla=False.")

        text_embeds = self.encode_text(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.model.logit_scale.exp()
        logits_per_image = logits_per_text.t()
        return logits_per_image, classhead


class Trainer(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.delr = opt.delr
        self.claloss = opt.claloss

        self.model = CLIPModelLora(
            name=opt.clip,
            lora_r=opt.lora_r,
            lora_alpha=opt.lora_alpha,
            lora_dropout=opt.lora_dropout,
        )

        trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        total_params = sum(param.numel() for param in self.model.parameters())
        print(f"trainable params={trainable_params}/{total_params}")

        if self.isTrain:
            self.loss_fn = nn.BCEWithLogitsLoss()
            params = filter(lambda parameter: parameter.requires_grad, self.model.parameters())
            if opt.optim == "adam":
                self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            elif opt.optim == "adamw":
                self.optimizer = torch.optim.AdamW(
                    params,
                    lr=opt.lr,
                    betas=(opt.beta1, 0.999),
                    weight_decay=0.05,
                    eps=1e-8,
                )
            elif opt.optim == "sgd":
                self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=0.0)
            else:
                raise ValueError("optim should be [adam, adamw, sgd]")

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

        self.model.to(self.device)

        self.amp = getattr(opt, "amp", False)
        self.amp_dtype = getattr(opt, "amp_dtype", "bf16").lower()
        self.amp_enabled = self.amp and self.device.type == "cuda"
        if self.amp_enabled and self.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
            self.amp_dtype = "fp16"
            if getattr(opt, "rank", 0) == 0:
                print("bf16 is not supported on this GPU; falling back to fp16.")
        elif self.amp and not self.amp_enabled and getattr(opt, "rank", 0) == 0:
            print("AMP requested but CUDA is unavailable; running in fp32.")

        self.autocast_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.autocast_dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
        scaler_enabled = self.amp_enabled and self.amp_dtype == "fp16"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    def adjust_learning_rate(self, min_lr=1e-6):
        for group in self.optimizer.param_groups:
            group["lr"] *= self.delr
            if group["lr"] < min_lr:
                return False
        self.lr = self.optimizer.param_groups[0]["lr"]
        print("*" * 25)
        print(f"Changing lr to {self.lr} with delr {self.delr}")
        print("*" * 25)
        return True

    def get_current_lr(self):
        if hasattr(self, "optimizer"):
            return self.optimizer.param_groups[0]["lr"]
        return self.lr

    def set_input(self, batch):
        self.path = batch[0]
        self.input = batch[1].to(self.device, non_blocking=True)
        self.text = batch[2]
        self.input_ids = batch[3].to(self.device, non_blocking=True)
        self.attention_mask = batch[4].to(self.device, non_blocking=True)
        self.label = batch[5].to(self.device, non_blocking=True).float()

    def forward(self):
        self.output, self.classhead = self.model(self.input, self.input_ids, self.attention_mask)

    @staticmethod
    def contrastive_loss(logits):
        targets = torch.arange(len(logits), device=logits.device)
        caption_loss = nn.functional.cross_entropy(logits, targets)
        image_loss = nn.functional.cross_entropy(logits.t(), targets)
        return (caption_loss + image_loss) / 2.0

    def optimize_parameters(self):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=self.autocast_device,
            dtype=self.autocast_dtype,
            enabled=self.amp_enabled,
        ):
            self.forward()
            logits = self.output.float()
            class_logits = self.classhead.float()
            self.loss1 = self.contrastive_loss(logits)
            self.loss2 = self.claloss * self.loss_fn(class_logits, self.label)
            self.loss = self.loss1 + self.loss2

        if self.scaler.is_enabled():
            self.scaler.scale(self.loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.loss.backward()
            self.optimizer.step()

        self.lr = self.get_current_lr()
        self.total_steps += 1
