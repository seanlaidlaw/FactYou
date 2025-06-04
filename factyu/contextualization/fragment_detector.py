import json

import spacy
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


class FragmentClassifier(torch.nn.Module):
    def __init__(self, base_model_dir):
        super().__init__()

        # Load the base model configuration
        self.encoder = AutoModel.from_pretrained(
            base_model_dir, ignore_mismatched_sizes=True
        )
        hidden = self.encoder.config.hidden_size
        self.classifier = torch.nn.Linear(hidden + 2, 2)

    def forward(self, input_ids, attention_mask, flags):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        feats = torch.cat([pooled, flags.float()], dim=1)
        return self.classifier(feats)


class SentenceFragmentDetector:
    def __init__(self, model_dir="sentence_frag_chkpt/best_fragment_model"):
        # Load model artifacts
        self.tok = AutoTokenizer.from_pretrained(
            "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
        )
        self.state = torch.load(f"{model_dir}/pytorch_model.bin", map_location="cpu")
        self.thr = json.load(open(f"{model_dir}/threshold.json"))["best_threshold"]

        # Initialize model
        self.model = FragmentClassifier(
            "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
        )
        self.model.load_state_dict(self.state)
        self.model.eval()

        # Initialize spaCy
        self.nlp = spacy.load("en_core_web_sm")
        self.DEMO = {"this", "that", "these", "those", "it", "they"}

    def make_flags(self, text):
        doc = self.nlp(text)
        root = [t for t in doc if t.dep_ == "ROOT"]
        miss_root = int(not root or root[0].pos_ not in {"VERB", "AUX"})
        orphan = 0
        for tok in doc:
            if tok.text.lower() in self.DEMO and tok.pos_ == "PRON":
                left = tok.nbor(-1) if tok.i else None
                if left is None or left.pos_ not in {"NOUN", "PROPN", "PRON"}:
                    orphan = 1
                    break
        return miss_root, orphan

    def is_standalone(self, sentence: str):
        """
        Determine if a sentence is complete/standalone or needs contextualization.

        Args:
            sentence (str): The sentence to analyze

        Returns:
            tuple: (is_complete, confidence_score)
                - is_complete (bool): True if the sentence is complete/standalone
                - confidence_score (float): Probability score for completeness
        """
        m_flag, d_flag = self.make_flags(sentence)
        flags = torch.tensor([[m_flag, d_flag]])
        enc = self.tok(sentence, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                flags=flags,
            )
            prob_complete = F.softmax(logits, dim=-1)[0, 1].item()

        return prob_complete >= self.thr, prob_complete
