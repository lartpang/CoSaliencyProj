from loss.CEs.BCELoss import cal_bce_loss
from loss.IOUs.IOULoss import cal_iou_loss, cal_weighted_iou_loss
from loss.LPs.L12Loss import cal_mae_loss, cal_mse_loss
from loss.SSIMs.SSIM import cal_ssim_loss

supported_loss = dict(
    bce=cal_bce_loss,
    iou=cal_iou_loss,
    weighted_iou=cal_weighted_iou_loss,
    mae=cal_mae_loss,
    mse=cal_mse_loss,
    ssim=cal_ssim_loss,
)


def get_loss_combination_with_cfg(loss_cfg: dict) -> dict:
    loss_combination = {}
    for loss_name, with_loss in loss_cfg.items():
        if with_loss:
            if loss_func := supported_loss.get(loss_name):
                loss_combination[loss_name] = loss_func
            else:
                raise Exception(f"{loss_name} is not be supported!")
    return loss_combination
