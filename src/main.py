from src.trainings.segmentation import train_segmentation
from src.trainings.centering import optimize_centers
from src.trainings.morph import train_morph
from src.trainings.instance_morph import train_morph_instant
import wandb

if __name__ == '__main__':


    # run_name = "seg_yesflip_norot_5000"
    run_name = 'general_not_aided_gradientl2_2_shift'
    # location = "/home/baris/Documents/brain-morphing"
    location = "/home/imreb/brain-morphing"
    # location = "/home/baris/Documents/work/brain-morphing"

    wandb.init(project="morph",
               entity="barisimre",
               name=run_name,
               notes="""(5.0 * loss_jacobian +
                5.0 * loss_gradient +
                loss_hematoma_decrease +
                5.0 * loss_skull +
                1.0 * loss_ventricle_overlap +
                loss_jeffrey +
                1.0 * loss_ssim +
                loss_ventricle_wrong_side

                )""")

    # train_segmentation(batch_size=2, run_name=run_name, location=location, hdf5_name="annotated.hdf5", dims=3, num_epochs=5000, loader_num_workers=8)
    # optimize_centers(location=location, run_name=run_name, batch_size=1, num_epochs=90, hdf5_target="all_data.hdf5")
    train_morph(run_name=run_name, num_epochs=1000, location=location, data_location="data/hdf5/ready_to_morph.hdf5", num_workers=8, log=True, mode='general')
    # train_morph_instant(run_name=run_name, num_epochs=50, location=location, data_location="data/hdf5/ready_to_morph.hdf5", num_workers=8, log=True, mode='instance_aided', lr=1e-3)

