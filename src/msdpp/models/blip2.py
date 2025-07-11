from pathlib import Path
from typing import TYPE_CHECKING

import datasets
import torch
from datasets import Dataset, DatasetDict
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForImageTextRetrieval

from msdpp import registry
from msdpp.base.model import BaseModel
from msdpp.schema import RetrievalResults

if TYPE_CHECKING:
    from transformers.models.blip_2.modeling_blip_2 import (
        Blip2ImageTextMatchingModelOutput,
    )

FEAT_DIM = 2


@registry.register_model("blip2")
class Blip2Model(BaseModel):
    def __init__(
        self,
        pretrained_model: str | Path,
        device: str | torch.device = "cuda:0",
        use_itm: bool = False,
        n_text_batch: int = 100,
    ) -> None:
        super().__init__(pretrained_model)  # set model_name

        self.model = Blip2ForImageTextRetrieval.from_pretrained(  # ty: ignore[possibly-unbound-attribute]
            pretrained_model, torch_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(pretrained_model)

        self.model = self.model.to(device)  # pyright: ignore[reportArgumentType]
        self.device = device
        self.use_itm = use_itm
        self.n_text_batch = n_text_batch

    def infer_image_text_similarity(
        self, image: Path | Image.Image | Tensor, text: str
    ) -> Tensor:
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(
            self.device, torch.float16
        )

        with torch.inference_mode():
            itm_out = self.model(**inputs, use_image_text_matching_head=self.use_itm)
            logits_per_image = torch.nn.functional.softmax(
                itm_out.logits_per_image, dim=1
            )
            return logits_per_image.softmax(dim=1)

    def extract_text_feats(self, text: str | list[str]) -> Tensor:
        if isinstance(text, str):
            text = [text]

        input_ids = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device, torch.float16)["input_ids"]
        query_embeds = self.model.embeddings(
            input_ids=input_ids,
        )
        text_outputs = self.model.qformer(
            query_embeds=query_embeds,
            query_length=0,
            attention_mask=None,
            return_dict=False,
        )
        question_embeds = text_outputs[0]

        return torch.nn.functional.normalize(
            self.model.text_projection(question_embeds[:, 0, :]), dim=-1
        )

    def txt_img_sim(self, text: str | Tensor, img_feats: Tensor) -> Tensor:
        if isinstance(text, Tensor):
            text_feats = text
        else:
            with torch.inference_mode():
                text_feats = self.extract_text_feats(text)
        if text_feats.dim() == 1:
            text_feats = text_feats.unsqueeze(0)

        norm_text_feats = torch.nn.functional.normalize(text_feats, dim=-1)
        norm_img_feats = torch.nn.functional.normalize(img_feats, dim=-1)

        logits, _ = torch.matmul(norm_img_feats, norm_text_feats.t()).max(dim=1)
        return logits.t()

    def img_sim(self, src_feats: Tensor, dst_feats: Tensor | None) -> Tensor:
        if dst_feats is None:
            dst_feats = src_feats

        if src_feats.dim() == FEAT_DIM:
            src_feats = src_feats.unsqueeze(0)
        if dst_feats.dim() == FEAT_DIM:
            dst_feats = dst_feats.unsqueeze(0)

        norm_src_feats = torch.nn.functional.normalize(src_feats, dim=-1)
        norm_dst_feats = torch.nn.functional.normalize(dst_feats, dim=-1)

        logits, _ = torch.einsum("ind,jnd->ijn", norm_src_feats, norm_dst_feats).max(
            dim=-1
        )
        return logits

    def infer_datasets(
        self, img_datasets: Dataset | DatasetDict, texts: list[str]
    ) -> RetrievalResults:
        t2i_sim_list = []
        img_embeds_list = []
        text_embeds_list = []

        if isinstance(img_datasets, DatasetDict):
            img_datasets = datasets.Dataset.from_dict(img_datasets)

        loader = self.prepare_loader(img_datasets)

        with torch.inference_mode():
            for i_text in range(0, len(texts), self.n_text_batch):
                text_batch = texts[i_text : i_text + self.n_text_batch]
                text_embeds_list.append(self.extract_text_feats(text_batch))
            text_embeds = torch.cat(text_embeds_list, 0)

            for images in tqdm(loader, desc="Extracting image features"):
                processed = self.processor(
                    images,
                    text=["test"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device, torch.float16)

                sim: Blip2ImageTextMatchingModelOutput = self.model(
                    **processed, use_image_text_matching_head=self.use_itm
                )

                logits_per_image = torch.matmul(sim.image_embeds, text_embeds.t())
                logits_per_text = logits_per_image.max(dim=1)[0].t()

                t2i_sim_list.append(logits_per_text.cpu())
                img_embeds_list.append(sim.image_embeds.cpu())

        t2i_sim = torch.cat(t2i_sim_list, 1)
        image_embeds = torch.cat(img_embeds_list, 0)

        return RetrievalResults(
            t2i_sim=t2i_sim, image_feats=image_embeds, text_feats=text_embeds.cpu()
        )
