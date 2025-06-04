import json
import os
from pathlib import Path

import spacy
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Get the package data directory
PACKAGE_DIR = Path(__file__).parent.parent
MODEL_DIR = PACKAGE_DIR / "data" / "models"


class FragmentClassifier(torch.nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        # Load the base model directly
        base_model_name = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
        self.encoder = AutoModel.from_pretrained(base_model_name)

        # Get the hidden size from the loaded model
        hidden_size = self.encoder.config.hidden_size

        # Create classifier head: hidden_size + 2 flags -> 2 classes
        self.classifier = torch.nn.Linear(hidden_size + 2, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        missing_verb_flag: torch.Tensor,
        dem_pron_flag: torch.Tensor,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        flags = torch.stack([missing_verb_flag, dem_pron_flag], dim=1).float()
        features = torch.cat([pooled, flags], dim=1)
        return self.classifier(features)


class FragmentDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize the fragment detector with the optimized model.

        Args:
            model_path: Optional path to a custom model directory. If None, uses the package's default model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use provided model path or default to package data
        if model_path is None:
            model_path = str(MODEL_DIR)

        # Load config
        with open(os.path.join(model_path, "config.json"), "r") as f:
            self.config = json.load(f)

        # Load threshold
        with open(os.path.join(model_path, "threshold.json"), "r") as f:
            self.threshold = json.load(f)["best_threshold"]

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Initialize model
        self.model = FragmentClassifier(model_path)

        # Load the saved state dict
        saved_state = torch.load(
            os.path.join(model_path, "pytorch_model.bin"), map_location=self.device
        )

        # Load state dict with strict=False to handle key mismatches
        # The classifier head should load correctly, and we'll ignore encoder mismatches
        missing_keys, unexpected_keys = self.model.load_state_dict(
            saved_state, strict=False
        )

        if missing_keys:
            print(
                f"Warning: Missing keys when loading model: {missing_keys[:5]}..."
            )  # Show first 5
        if unexpected_keys:
            print(
                f"Warning: Unexpected keys when loading model: {unexpected_keys[:5]}..."
            )  # Show first 5

        self.model.to(self.device)
        self.model.eval()

        # Load spaCy
        self.nlp = spacy.load("en_core_web_sm")
        self.demo_prons = {"this", "that", "these", "those", "it", "they"}

    def extract_spacy_flags(self, text: str):
        """Extract spaCy flags for fragment detection."""
        doc = self.nlp(text)
        # missing_verb_flag
        roots = [tok for tok in doc if tok.dep_ == "ROOT"]
        if roots:
            root = roots[0]
            missing_verb_flag = int(root.pos_ not in {"VERB", "AUX"})
        else:
            missing_verb_flag = 1
        # dem_pron_flag
        dem_pron_flag = 0
        for tok in doc:
            if tok.text.lower() in self.demo_prons and tok.pos_ == "PRON":
                left = tok.nbor(-1) if tok.i > 0 else None
                if left is None or left.pos_ not in {"NOUN", "PROPN", "PRON"}:
                    dem_pron_flag = 1
                    break
        return missing_verb_flag, dem_pron_flag

    def is_complete_sentence(self, text: str) -> bool:
        """
        Determine if the given text is a complete sentence.

        Args:
            text: The text to check

        Returns:
            bool: True if the text is a complete sentence, False otherwise
        """
        # Tokenize
        inputs = self.tokenizer(
            text, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        # Get spaCy flags
        missing_verb_flag, dem_pron_flag = self.extract_spacy_flags(text)
        flags = torch.tensor(
            [[missing_verb_flag, dem_pron_flag]], dtype=torch.float32
        ).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                missing_verb_flag=flags[:, 0],
                dem_pron_flag=flags[:, 1],
            )
            probs = torch.softmax(outputs, dim=-1)
            return bool(probs[0, 1] >= self.threshold)

    def make_flags(self, text):
        doc = self.nlp(text)
        root = [t for t in doc if t.dep_ == "ROOT"]
        miss_root = int(not root or root[0].pos_ not in {"VERB", "AUX"})
        orphan = 0
        for tok in doc:
            if tok.text.lower() in self.demo_prons and tok.pos_ == "PRON":
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
        enc = self.tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=512
        )

        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                missing_verb_flag=flags[:, 0],
                dem_pron_flag=flags[:, 1],
            )
            prob_complete = F.softmax(logits, dim=-1)[0, 1].item()

        return prob_complete >= self.threshold, prob_complete
