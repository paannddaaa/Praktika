import torch
import pytorch_lightning as pl
from lightning_train import LitFacesModule, NN2 #NN1

#CHECKPOINT_PATH = "D:\pyt3.10\pythonProject\FaceNet_clean_2\FaceNet_clean\saved_ckpts\sample-faces-nn1-epoch=74-val_loss=0.4018.ckpt"
CHECKPOINT_PATH ='D:\pyt3.10\pythonProject\FaceNet_clean_2\FaceNet_clean\saved_ckpts\sample-faces-nn2-epoch=77-val_loss=0.30.ckpt'

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    print("Чекпоинты")
    for key, value in checkpoint.items():
        if isinstance(value, (int, float, str)):
            print(f"  - {key}: {value}")
        else:
            print(f"  - {key}: {type(value)}, size: {len(value) if hasattr(value, '__len__') else 'N/A'}")

    lit_model = LitFacesModule.load_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        model=NN2()
    )
    lit_model = lit_model.to(DEVICE)
    print("У тебя получилось...")



if __name__ == "__main__":
    main()
