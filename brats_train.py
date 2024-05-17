import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
import torch
from dataset_brats import get_loader
import numpy as np

print_config()

from model_segmamba.mambaconvnext import SegMamba

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
set_determinism(seed=0)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

# class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
#     """
#     Convert labels to multi channels based on brats classes:
#     label 1 is the peritumoral edema
#     label 2 is the GD-enhancing tumor
#     label 3 is the necrotic and non-enhancing tumor core
#     The possible classes are TC (Tumor core), WT (Whole tumor)
#     and ET (Enhancing tumor).

#     """

#     def __call__(self, data):
#         d = dict(data)
#         for key in self.keys:
#             result = []
#             # merge label 2 and label 3 to construct TC
#             result.append(torch.logical_or(d[key] == 2, d[key] == 3))
#             # merge labels 1, 2 and 3 to construct WT
#             result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
#             # label 2 is ET
#             result.append(d[key] == 2)
#             d[key] = torch.stack(result, axis=0).float()
#         return d

# train_transform = Compose(
#     [
#         # load 4 Nifti images and stack them together
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys="image"),
#         EnsureTyped(keys=["image", "label"]),
#         ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "label"],
#             pixdim=(1.0, 1.0, 1.0),
#             mode=("bilinear", "nearest"),
#         ),
#         RandSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 128], random_size=False),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
#         RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
#         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#         RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
#         RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
#     ]
# )
# val_transform = Compose(
#     [
#         LoadImaged(keys=["image", "label"]),
#         EnsureChannelFirstd(keys="image"),
#         EnsureTyped(keys=["image", "label"]),
#         ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
#         Orientationd(keys=["image", "label"], axcodes="RAS"),
#         Spacingd(
#             keys=["image", "label"],
#             pixdim=(1.0, 1.0, 1.0),
#             mode=("bilinear", "nearest"),
#         ),
#         NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
#     ]
# )

# train_ds = DecathlonDataset(
#     root_dir=root_dir,
#     task="Task01_BrainTumour",
#     transform=train_transform,
#     section="training",
#     download=True,
#     cache_rate=0.0,
#     num_workers=4,
# )
# train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
# val_ds = DecathlonDataset(
#     root_dir=root_dir,
#     task="Task01_BrainTumour",
#     transform=val_transform,
#     section="validation",
#     download=False,
#     cache_rate=0.0,
#     num_workers=4,
# )
# val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

train_loader, val_loader, train_ds_len, val_ds_len = get_loader()

max_epochs = 200
val_interval = 1
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")





#t1 = torch.rand(1, 4, 128, 128, 128).cuda()


model = SegMamba(in_chans=4,
                 out_chans=3,
                 depths=[2,2,2,2],
                 feat_size=[48, 96, 192, 384]).cuda()

# out = model(t1)

# print(out.shape)

model = model.to(device)

####### use pretrained weights #########
use_pretrained = False
pretrained_path = os.path.normpath("pretrained_weights/model_bestValRMSE_6feb_val_loss0.20440135553479194.pt")

# Load ViT backbone weights into UNETR
if use_pretrained is True:
    print("Loading Weights from the Path {}".format(pretrained_path))
    vit_dict = torch.load(pretrained_path)
    vit_weights = vit_dict["state_dict"]

    # Remove items of vit_weights if they are not in the ViT backbone (this is used in UNETR).
    # For example, some variables names like conv3d_transpose.weight, conv3d_transpose.bias,
    # conv3d_transpose_1.weight and conv3d_transpose_1.bias are used to match dimensions
    # while pretraining with ViTAutoEnc and are not a part of ViT backbone.
    model_dict = model.vit.state_dict()

    vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
    model_dict.update(vit_weights)
    model.vit.load_state_dict(model_dict)
    del model_dict, vit_weights, vit_dict
    print("Pretrained Weights Succesfully Loaded !")

elif use_pretrained is False:
    print("No weights were loaded, all weights being used are randomly initialized!")


# # Freezing Encoder
# for param in model.vit.parameters():
#     param.requires_grad = False

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []
mean_dice_max = 0.0

total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(
            f"{step}/{train_ds_len // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}"
            f", step time: {(time.time() - step_start):.4f}"
        )
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        dice_tc_list = []
        dice_et_list = []
        dice_wt_list = []
        run_acc = AverageMeter()
        v_step = 0
        all_dice = []
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                
                dice_acc.reset()
                dice_acc(y_pred = val_outputs, y = val_labels )
                acc, not_nans = dice_acc.aggregate()
                run_acc.update(acc.cpu().numpy(), n = not_nans.cpu().numpy())
                dice_tc = run_acc.avg[0]
                dice_wt = run_acc.avg[1]
                dice_et = run_acc.avg[2]
                #val_avg_acc = np.mean(run_acc)
                print(
                    "Val {}/{} {}/{}".format(epoch, max_epochs, v_step, len(val_loader)),
                    ", dice_tc:",
                    dice_tc,
                    ", dice_wt:",
                    dice_wt,
                    ", dice_et:",
                    dice_et,
                )
                v_step += 1
                dice_tc_list.append(dice_tc)
                dice_wt_list.append(dice_wt)
                dice_et_list.append(dice_et)
            mean_dice = (np.mean(dice_tc_list)+np.mean(dice_wt_list)+np.mean(dice_et_list))/3
            print('Epoch: ' + str(epoch + 1), 'Mean Dice Score: ' + str(mean_dice))
            if mean_dice > mean_dice_max:
                mean_dice_max = mean_dice
                torch.save(model.state_dict(), 'saved_model/' + 'epoch_' + str(epoch) + '_dice_' + str(mean_dice_max) + '.pth')
                print('Checkpoint_' + 'epoch_' + str(epoch) + '_dice_' + str(mean_dice_max) + '.pth' + ' saved')


            #     dice_metric(y_pred=val_outputs, y=val_labels)
            #     dice_metric_batch(y_pred=val_outputs, y=val_labels)

            # metric = dice_metric.aggregate().item()
            # metric_values.append(metric)
            # metric_batch = dice_metric_batch.aggregate()
            # metric_tc = metric_batch[0].item()
            # metric_values_tc.append(metric_tc)
            # metric_wt = metric_batch[1].item()
            # metric_values_wt.append(metric_wt)
            # metric_et = metric_batch[2].item()
            # metric_values_et.append(metric_et)
            # dice_metric.reset()
            # dice_metric_batch.reset()

            # if metric > best_metric:
            #     best_metric = metric
            #     best_metric_epoch = epoch + 1
            #     best_metrics_epochs_and_time[0].append(best_metric)
            #     best_metrics_epochs_and_time[1].append(best_metric_epoch)
            #     best_metrics_epochs_and_time[2].append(time.time() - total_start)
            #     torch.save(
            #         model.state_dict(),
            #         os.path.join('saved_model/'+'epoch_'+str(epoch)+'_dice_'+str(metric)+'best_metric_model.pth'),
            #     )
            #     print("saved new best metric model")
            # print(
            #     f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
            #     f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
            #     f"\nbest mean dice: {best_metric:.4f}"
            #     f" at epoch: {best_metric_epoch}"
            # )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

# import matplotlib.pyplot as plt

# # First set of plots
# plt.figure("Epoch Analysis", (12, 6))

# # Plot for Epoch Average Loss
# plt.subplot(1, 2, 1)
# plt.title("Epoch Average Loss")
# x = [i + 1 for i in range(len(epoch_loss_values))]
# y = epoch_loss_values
# plt.xlabel("epoch")
# plt.plot(x, y, color="red")
# plt.savefig("Epoch_Average_Loss.png")  # Save the first plot

# # Plot for Val Mean Dice
# plt.subplot(1, 2, 2)
# plt.title("Val Mean Dice")
# x = [val_interval * (i + 1) for i in range(len(metric_values))]
# y = metric_values
# plt.xlabel("epoch")
# plt.plot(x, y, color="green")
# plt.savefig("Val_Mean_Dice.png")  # Save the second plot

# plt.show()

# # Second set of plots
# plt.figure("Detailed Val Mean Dice Analysis", (18, 6))

# # Plot for Val Mean Dice TC
# plt.subplot(1, 3, 1)
# plt.title("Val Mean Dice TC")
# x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
# y = metric_values_tc
# plt.xlabel("epoch")
# plt.plot(x, y, color="blue")
# plt.savefig("Val_Mean_Dice_TC.png")  # Save the first plot in the second figure

# # Plot for Val Mean Dice WT
# plt.subplot(1, 3, 2)
# plt.title("Val Mean Dice WT")
# x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
# y = metric_values_wt
# plt.xlabel("epoch")
# plt.plot(x, y, color="brown")
# plt.savefig("Val_Mean_Dice_WT.png")  # Save the second plot in the second figure

# # Plot for Val Mean Dice ET
# plt.subplot(1, 3, 3)
# plt.title("Val Mean Dice ET")
# x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
# y = metric_values_et
# plt.xlabel("epoch")
# plt.plot(x, y, color="purple")
# plt.savefig("Val_Mean_Dice_ET.png")  # Save the third plot in the second figure

# plt.show()






