import matplotlib.pyplot as plt
import torch
import wandb


def plot(img, cmap="gray"):
    x, y = img.shape
    plt.axvline(x=x // 2)
    plt.imshow(img, cmap=cmap)
    plt.show()


def vis_to_wandb_segmentation(img, output, mask, names, loss, epoch, save=False, save_path=None, use_wandb=True):
    with torch.no_grad():
        # Get the predicted masks from the output logits
        _, predicted_masks = torch.max(output, dim=1)
        _, mask = torch.max(mask, dim=1)

        # Convert the inputs, labels, and predicted masks to numpy arrays
        inputs = img.cpu().detach().numpy()
        labels = mask.cpu().detach().numpy()
        predicted_masks = predicted_masks.cpu().detach().numpy()

        # From the batch
        i = 0
        name = names[i]
        image = inputs[i, 0]
        label = labels[i]
        predicted_mask = predicted_masks[i]

        # Create a figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        cmap = plt.cm.get_cmap('tab10', 4)

        # Plot the input image in the first subplot
        axs[0].imshow(image, cmap="gray")
        axs[0].set_title(f'Input image')

        # Plot the label in the second subplot
        axs[1].imshow(label, cmap=cmap, vmin=0, vmax=3)
        axs[1].set_title(f'Label')

        # Plot the predicted mask in the third subplot
        axs[2].imshow(predicted_mask, cmap=cmap, vmin=0, vmax=3)
        axs[2].set_title(f'Predicted mask')

        fig.suptitle(f"{name}  loss: {loss}")

        if save:
            assert save_path is not None
            plt.savefig(f'{save_path}/{epoch}.png', dpi=200)
        if use_wandb:
            wandb.log({"plot": fig})
        else:
            plt.show()
        plt.close(fig)


def vis_to_wandb_segmentation_3d(img, output, mask, names, loss, epoch, save=False, save_path=None, use_wandb=True):
    with torch.no_grad():
        # Get the predicted masks from the output logits
        _, predicted_masks = torch.max(output, dim=1)
        _, mask = torch.max(mask, dim=1)

        # Convert the inputs, labels, and predicted masks to numpy arrays
        inputs = img.cpu().detach().numpy()
        labels = mask.cpu().detach().numpy()
        predicted_masks = predicted_masks.cpu().detach().numpy()

        # From the batch
        i = 0
        name = names[i]
        image = inputs[i, 0]
        label = labels[i]
        predicted_mask = predicted_masks[i]

        halves = [x // 2 for x in image.shape]

        # Create a figure with 3 subplots
        fig, axs = plt.subplots(3, 3, figsize=(15, 15))

        cmap = plt.cm.get_cmap('tab10', 4)

        # Plot the input image in the first subplot
        axs[2, 0].imshow(image[halves[0]], cmap="gray", aspect='auto')
        axs[1, 0].imshow(image[:, halves[1]], cmap="gray", aspect='auto')
        axs[0, 0].imshow(image[:, :, halves[2]], cmap="gray")

        axs[0, 0].set_title(f'Input image')

        # Plot the label in the second subplot
        axs[2, 1].imshow(label[halves[0]], cmap=cmap, vmin=0, vmax=3, aspect='auto')
        axs[1, 1].imshow(label[:, halves[1]], cmap=cmap, vmin=0, vmax=3, aspect='auto')
        axs[0, 1].imshow(label[:, :, halves[2]], cmap=cmap, vmin=0, vmax=3)

        axs[0, 1].set_title(f'Label')

        # Plot the predicted mask in the third subplot
        axs[2, 2].imshow(predicted_mask[halves[0]], cmap=cmap, vmin=0, vmax=3, aspect='auto')
        axs[1, 2].imshow(predicted_mask[:, halves[1]], cmap=cmap, vmin=0, vmax=3, aspect='auto')
        axs[0, 2].imshow(predicted_mask[:, :, halves[2]], cmap=cmap, vmin=0, vmax=3)

        axs[0, 2].set_title(f'Predicted mask')

        fig.suptitle(f"{name}  loss: {loss}")

        if save:
            assert save_path is not None
            plt.savefig(f'{save_path}/{epoch}.png', dpi=200)

        if use_wandb:
            wandb.log({"plot": fig})
        else:
            plt.imshow(fig)
        plt.close(fig)


def detailed_morph(img, morphed, d_field, slice=None, line=True, name="unnamed", loss=0.0, save=False, save_location="",
                   cmap="gray",cmap_d="Reds", use_wandb=False):
    if slice is None:
        slice_z = int(min(torch.tensor(img.shape[-3:]) / 2))
        slice_x = int(max(torch.tensor(img.shape[-3:]) / 2))
    else:
        slice_z = slice
        slice_x = slice

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # print(d_field.shape)

    img = img[0, 0].cpu().detach()
    morphed = morphed[0, 0].cpu().detach()
    d_field = d_field[0].cpu().detach()
    d_field = torch.sum(d_field, dim=0)

    # print(img.shape)

    slices_img = [torch.rot90(img[slice_x, :, :]), img[:, :, slice_z]]
    slices_morphed = [torch.rot90(morphed[slice_x, :, :]), morphed[:, :, slice_z]]
    slices_deformation = [torch.rot90(d_field[slice_x, :, :]), d_field[:, :, slice_z]]

    axs[1, 0].imshow(slices_img[0], cmap=cmap)
    axs[1, 1].imshow(slices_morphed[0], cmap=cmap)
    axs[1, 2].imshow(slices_deformation[0], cmap=cmap_d)

    axs[0, 0].imshow(slices_img[1], cmap=cmap)
    axs[0, 1].imshow(slices_morphed[1], cmap=cmap)
    axs[0, 2].imshow(slices_deformation[1], cmap=cmap_d)
    # for i, ax in enumerate(axs.flat):
    #     im = ax.imshow(slices[i], cmap=cmap)
    #     if line:
    #         ax.axvline(x=slices[i].shape[1] // 2)

    if name != "unnamed":
        fig.suptitle(f"Iteration: {name}, Loss: {loss:.3f}")
    if save:
        fig.savefig(f"{save_location}/{name}.png", dpi=200)

    if use_wandb:
        wandb.log({name: fig})

    plt.show(block=False)
    plt.close()


def plot_from3d(img, slice=200, line=True, mask=None):
    s = None
    if len(img.shape) == 5:
        s = img[0, 0, :, :, slice].cpu().detach()
    elif len(img.shape) == 4:
        s = img[0, :, :, slice].cpu().detach()

    elif len(img.shape) == 3:
        s = img[:, :, slice].cpu().detach()

    m = None
    if mask is not None:
        if len(mask.shape) == 5:
            m = mask[0, 0, :, :, slice].cpu().detach()
        elif len(mask.shape) == 4:
            m = mask[0, :, :, slice].cpu().detach()

        elif len(img.shape) == 3:
            m = mask[:, :, slice].cpu().detach()

    x = s.shape[1]
    plt.imshow(s, cmap="gray")
    if mask is not None:
        plt.imshow(m == 1, alpha=0.4)
        print(mask.unique())
    if line:
        plt.axvline(x=x // 2)
    plt.show(block=False)
    plt.close()


def detailed_plot_from3d(img, slice=None, line=True, name="unnamed", loss=0.0, save=False, save_location="",
                         cmap="gray", use_wandb=False):
    if slice is None:
        slice_z = int(min(torch.tensor(img.shape[-3:]) / 2))
        slice_x = int(max(torch.tensor(img.shape[-3:]) / 2))
    else:
        slice_z = slice
        slice_x = slice

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    if len(img.shape) == 5:
        img = img[0, 0]
    elif len(img.shape) == 4:
        img = img[0]

    try:
        img = img.cpu().detach()
    except AttributeError:
        img = torch.tensor(img)

    slices = [torch.rot90(img[slice_x, :, :]), img[:, :, slice_z]]
    titles = ['X-axis slice', 'Z-axis slice']

    for i, ax in enumerate(axs.flat):
        im = ax.imshow(slices[i], cmap=cmap)
        if line:
            ax.axvline(x=slices[i].shape[1] // 2)

        ax.set_title(titles[i])

    if name != "unnamed":
        fig.suptitle(f"Iteration: {name}, Loss: {loss:.3f}")
    if save:
        fig.savefig(f"{save_location}/{name}.png", dpi=200)

    plt.colorbar(im, ax=axs.ravel().tolist())

    if use_wandb:
        wandb.log({name: fig})

    plt.show(block=False)
    plt.close()
