# main_train.py
from configs.train_config import TrainConfig
from datasets.dataset_loader import TeacherDataset
from models.teacher import load_teacher
from models.student import build_scheduler
from train.train_scheduler_only import train_scheduler

def main():
    cfg = TrainConfig()
    dataset = TeacherDataset("data/")
    teacher_models = load_teacher()
    scheduler = build_scheduler(cond_dim=768)

    train_scheduler(dataset, teacher_models, scheduler, cfg)

if __name__ == "__main__":
    main()
