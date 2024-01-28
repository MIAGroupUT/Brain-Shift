from src.trainings.segmentation import train_segmentation
import wandb

if __name__ == '__main__':
    run_name = "Debug"
    # location = "/home/baris/Documents/brain-morphing"
    location = "/home/imreb/brain-morphing"

    wandb.init(project="Debug",
               entity="barisimre",
               name=run_name)

    train_segmentation(batch_size=8, run_name=run_name, location=location)
