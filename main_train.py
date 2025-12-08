# main_train.py
import torch
import sys
sys.path.append('/content/drive/MyDrive/Adversarial-Diffusion-Distillation')
from configs.model_config import ModelConfig
from configs.train_config import TrainConfig
from models.teacher import load_teacher_model
from models.student import RobustLearnedScheduler
from diffusion.sampler_student import DifferentiableDiffusionHandler
from train.train_step import train_one_step

def main():
    # 1. Setup
    print(f"Starting Project on {TrainConfig.DEVICE}")
    pipe = load_teacher_model(TrainConfig.DEVICE)
    diff_handler = DifferentiableDiffusionHandler(pipe)
    
    # 2. Initialize Student
    student = RobustLearnedScheduler(num_steps=ModelConfig.STUDENT_STEPS).to(TrainConfig.DEVICE)
    optimizer = torch.optim.AdamW(student.parameters(), lr=TrainConfig.LEARNING_RATE)
    
    # 3. Dummy Data (Replace with datasets/dataset_loader.py in real version)
    prompts = [
        "A futuristic city with flying cars",
        "A cute corgi running in a field"
    ]
    
    # Pre-encode prompts
    encoded_prompts = []
    for p in prompts:
        inputs = pipe.tokenizer(p, return_tensors="pt", padding="max_length", truncation=True).to(TrainConfig.DEVICE)
        with torch.no_grad():
            encoded_prompts.append(pipe.text_encoder(inputs.input_ids)[0])

    # 4. Training Loop
    for epoch in range(TrainConfig.EPOCHS):
        total_loss = 0
        for text_emb in encoded_prompts:
            loss = train_one_step(
                student, 
                diff_handler, 
                optimizer, 
                pipe, 
                text_emb, 
                ModelConfig.STUDENT_STEPS
            )
            total_loss += loss
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(encoded_prompts):.4f}")

    # 5. Save (Optional)
    torch.save(student.state_dict(), "student_scheduler.pth")
    print("Done!")

if __name__ == "__main__":
    main()
