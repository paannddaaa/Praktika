import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import CelebaTripletDataset
from lightning_train import LitFacesModule, NN2
import random
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

def calculate_distance(vector1, vector2, metric="euclidean"):
    """Вычисляет расстояние между двумя векторами."""
    if metric == "euclidean":
        return torch.norm(vector1 - vector2, p=2, dim=1).cpu().numpy()
    elif metric == "cosine":
        return (1 - torch.nn.functional.cosine_similarity(vector1, vector2, dim=1)).cpu().numpy()

if __name__ == '__main__':
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели
    CHECKPOINT_PATH = r"D:\pyt3.10\pythonProject\FaceNet_clean_2\FaceNet_clean\saved_ckpts\sample-faces-nn2-epoch=77-val_loss=0.30.ckpt"
    lit_model = LitFacesModule.load_from_checkpoint(CHECKPOINT_PATH, model=NN2())
    lit_model.to(device)
    lit_model.eval()

    # Подготовка данных
    root_dir = r"D:\Celeba\img_align_celeba\img_align_celeba"
    csv_path = r"D:\pyt3.10\pythonProject\CelebaTriplets.csv"
    transform = Compose([
        Resize(height=220, width=220),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    dataset = CelebaTripletDataset(csv_path, root_dir, transforms=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)


    def calculate_distance(vector1, vector2, metric="euclidean"):
        """Вычисляет расстояние между двумя векторами."""
        if metric == "euclidean":
            return torch.norm(vector1 - vector2, p=2, dim=1).cpu().numpy()
        elif metric == "cosine":
            return (1 - torch.nn.functional.cosine_similarity(vector1, vector2, dim=1)).cpu().numpy()


    if __name__ == '__main__':
        # Устройство
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загрузка модели
        CHECKPOINT_PATH = r"D:\pyt3.10\pythonProject\FaceNet_clean_2\FaceNet_clean\saved_ckpts\sample-faces-nn2-epoch=77-val_loss=0.30.ckpt"
        lit_model = LitFacesModule.load_from_checkpoint(CHECKPOINT_PATH, model=NN2())
        lit_model.to(device)
        lit_model.eval()

        # Подготовка данных
        root_dir = r"D:\Celeba\img_align_celeba\img_align_celeba"
        csv_path = r"D:\pyt3.10\pythonProject\CelebaTriplets.csv"
        transform = Compose([
            Resize(height=220, width=220),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        dataset = CelebaTripletDataset(csv_path, root_dir, transforms=transform)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

        # Выбираем случайное изображение
        random_index = random.randint(0, len(dataset) - 1)
        random_sample = dataset[random_index]
        random_image = random_sample['base_anchor'].unsqueeze(0).to(device)  # Добавляем batch размерность

        # Вычисляем вектор для случайного изображения
        with torch.no_grad():
            random_vector = lit_model.model(random_image)

        # Списки расстояний
        same_face_distances = []
        different_face_distances = []

        # Генерация расстояний
        for i, batch in enumerate(loader):
            base_anchor = batch['base_anchor'].to(device)
            person_ids = batch['person_id']

            with torch.no_grad():
                vectors = lit_model.model(base_anchor)

                # Вычисляем расстояния до случайного изображения
                distances = calculate_distance(random_vector, vectors)

                # Разделяем расстояния
                for j, distance in enumerate(distances):
                    if person_ids[j] == random_sample['person_id']:
                        same_face_distances.append(distance)
                    else:
                        different_face_distances.append(distance)

            # Прерываем после обработки 50 батчей (оптимизация)
            if i >= 50:
                break

        # Проверка на пустые списки
        if len(same_face_distances) == 0:
            print("Ошибка: массив same_face_distances пуст!")
        if len(different_face_distances) == 0:
            print("Ошибка: массив different_face_distances пуст!")

        # Построение гистограммы, если данные не пустые
        if same_face_distances and different_face_distances:
            plt.figure(figsize=(10, 6))
            bins = np.linspace(0, max(np.max(same_face_distances), np.max(different_face_distances)), 30)

            plt.hist(same_face_distances, bins=bins, alpha=0.7, label="Одинаковые лица", color='blue')
            plt.hist(different_face_distances, bins=bins, alpha=0.7, label="DРазные лица", color='red')
            plt.axvline(np.mean(same_face_distances), color='blue', linestyle='dashed', linewidth=1,
                        label="Среднее по схожим лицам")
            plt.axvline(np.mean(different_face_distances), color='red', linestyle='dashed', linewidth=1,
                        label="Среднее по отличающимся лицам")

            plt.xlabel("Расстояние")
            plt.ylabel("Количество попаданий")
            plt.legend()
            plt.title("Гистограмма разницы")
            plt.show()
        else:
            print("Гистограмма не может быть построена из-за пустых данных.")