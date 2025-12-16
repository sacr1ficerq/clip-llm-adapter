from torch.utils.data import Dataset
import random


# This function is partially written by LLM, because I am not famimiliar with those kinds of datasets
class Flickr8kCaptionDataset(Dataset):
    """
    Returns: (PIL_image, caption_str)
    Assumes ds split items contain an 'image' field and some caption text field.
    The jxie/flickr8k dataset is an 8k-row captioning dataset [web:2].
    """

    def __init__(self, hf_split, caption_key_candidates=("caption", "text", "sentence", "caption_0")):
        self.ds = hf_split
        self.caption_key_candidates = caption_key_candidates

        # pick a caption key that exists
        sample = self.ds[0]
        self.caption_key = None
        for k in caption_key_candidates:
            if k in sample:
                self.caption_key = k
                break
        if self.caption_key is None:
            # last resort: try to find any string field
            for k, v in sample.items():
                if isinstance(v, str):
                    self.caption_key = k
                    break

        assert self.caption_key is not None, f"no caption in dataset: {list(sample.keys())}"
        assert "image" in sample, f"no image in sample: {list(sample.keys())}"

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        image = ex["image"]          # PIL image from HF datasets
        caption = ex[self.caption_key]

        # If dataset stores multiple captions per image as list, pick one at random
        if isinstance(caption, (list, tuple)):
            caption = random.choice(caption)

        return image, str(caption)


def collate_fn_vision(batch, image_processor, tokenizer, device, max_txt_len=64):
    images, captions = zip(*batch)  # batch: list[(PIL_image, caption_str)]

    # CLIP preprocessing -> tensor
    img_inputs = image_processor(images=list(images), return_tensors="pt")
    pixel_values = img_inputs["pixel_values"]  # (B, 3, H, W)

    # tokenize captions for teacher forcing
    tok = tokenizer(
        list(captions),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_txt_len,
    )

    return {
        "pixel_values": pixel_values.to(device),  # (B, 3, H, W)
        "input_ids": tok["input_ids"].to(device),  # (B, L)
        "attention_mask": tok["attention_mask"].to(device),  # (B, L)
        "captions": list(captions),  # list[str]
    }
