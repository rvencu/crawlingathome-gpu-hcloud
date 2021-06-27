import clip
import datasets
import torch
from anyascii import anyascii
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
datasets.set_caching_enabled(False)

vmem = torch.cuda.get_device_properties(0).total_memory if device == "cuda" else 0
batch_size = 128 * int(vmem/1800000000) if device == "cuda" else 8
print(f"batch size = {batch_size}")

class CLIP:
    def __init__(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.categories = self.model.encode_text(clip.tokenize(["neutral","selfie", "illustration, drawing", "toys, play, kids, children", "teddy bear, puppet", "animal, bird, mammal, insect" "fashion, clothes", "logo, commercial, ad, advertisement", "drawing, painting","anime, cartoon","comedy, fun","romance, love story","thriller, suspense, crime story","action, action movie", "horror, monster movie", "documentary", "news, journalism", "entertainment", "talk show", "porn, sex, sperm, nipples, breats, tits, boops, penis, dick, cock, clitoris, vagina, fuck, lust, horny, sexual, lick, licking",  "porn, sex, sperm, nipples", "porn, sex, sperm, penis, dick, cock", "nipples, breats, tits, boops, sexy", "penis, dick, cock", "clitoris, vagina", "sex, fuck, lust, horny, sexual, lick, licking", "porn, sex, sexy","sexy, hot","sperm, skin","lust, horny, sexual","lick, licking, body", "anime, hentai, sexy", "cartoon, sexy, sex", "hentai", "anime, sexy, breasts", "hentai"]).to(device))
        self.underaged_categories = self.model.encode_text(clip.tokenize(["teenager, teen", "kid, child, teenager, teen, baby or toddler, underaged, little girl, little boy", "kid, child, little girl, little boy", "baby, toddler","adult, woman, man, grownup, grown person,full-aged of legal age","full-aged, of legal age, adult","woman, man","adult, woman, man, grownup, grown person,full-aged of legal age"]).to(device))
        self.animal_categories = self.model.encode_text(clip.tokenize(["lifeless object, thing", "thing, object", "material", "furniture","wall", "house", "tree", "wood","ground","industry", "table", "bed", "tool", "dress, clothes", "door", "chair", "rock, stone", "human", "man", "woman", "man, woman", "animal","cat","dog", "cow", "pig", "goat", "sheep", "elephant", "horse", "horse, elephant, pig, dog, cat, sheep, goat, animal", "life", "wildlife"]).to(device))


    def similarity_imgalt(self, batch):
        similarity = []
        images = [
            self.preprocess(Image.open(path)).unsqueeze(0).to(device)
            for path in batch["PATH"]
        ]
        max_texts = [anyascii(text)[:77] for text in batch["TEXT"]]
        texts = clip.tokenize(max_texts).to(device)

        with torch.no_grad():
            image_features = self.model.encode_image(
                torch.cat(images)
            ).float()
            text_features = self.model.encode_text(texts).float()

        for image_feat, text_feat in zip(image_features, text_features):
            similarity.append(
                float(
                    self.cosine_similarity(
                        torch.reshape(text_feat, (1, 512)),
                        torch.reshape(image_feat, (1, 512)),
                    )
                )
            )

        batch["similarity"] = similarity
        batch["image_features"] = image_features.detach().cpu().numpy()
        return batch

    def preprocess_images(self, df):
        im_dataset = datasets.Dataset.from_pandas(df)
        im_dataset = im_dataset.map(self.similarity_imgalt, batched=True, batch_size=batch_size)
        return im_dataset["image_features"], im_dataset["similarity"]

    def prob(self, image_features, text_features):
        with torch.no_grad():
            image_features = torch.as_tensor(image_features).to(device)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            similarity = (100.0 * image_features.float() @ text_features.T.float()).softmax(dim=-1)
            _, indices = similarity.topk(2)
            return indices