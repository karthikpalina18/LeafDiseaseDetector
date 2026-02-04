import gdown

# ✅ Google Drive Direct Download Link
MODEL_URL = "https://drive.google.com/uc?id=1JP-OctUi67Tuq6ulxUtqNSTCNw0x6iKf"

# ✅ Download Model
print("⬇️ Downloading model from Google Drive...")

gdown.download(MODEL_URL, "leaf_model_resnet50_final.h5", quiet=False)

print("✅ Model downloaded successfully as leaf_model_resnet50_final.h5")
