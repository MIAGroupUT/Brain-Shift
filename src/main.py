from src.trainings.segmentation import train_segmentation
import wandb

if __name__ == '__main__':
    run_name = "segmentation_3d_rotatefirst"
    # location = "/home/baris/Documents/brain-morphing"
    location = "/home/imreb/brain-morphing"

    wandb.init(project="annotations",
               entity="barisimre",
               name=run_name)

    train_segmentation(batch_size=2, run_name=run_name, location=location, slice_thickness="large", dims=3)
