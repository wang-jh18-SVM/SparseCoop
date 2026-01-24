from inspect import signature

import os
import mmcv
import torch

from mmcv.runner import force_fp32, auto_fp16
from .utils.misc import locations
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmdet.models import DETECTORS, BaseDetector, build_backbone, build_head, build_neck
from .grid_mask import GridMask

try:
    from ..ops import feature_maps_format

    DAF_VALID = True
except:
    DAF_VALID = False

__all__ = ["SparseCoop"]


@DETECTORS.register_module()
class SparseCoop(BaseDetector):

    def __init__(
        self,
        img_backbone,
        head=None,
        img_neck=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        depth_branch=None,
        # Far3D
        img_roi_head=None,
        # Coop
        save_track_query=False,
        save_track_query_file_root=None,
        save_topK=100,
        save_threshold=None,
        save_dn_for_coop=False,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_img_roi_head=False,
    ):
        super(SparseCoop, self).__init__(init_cfg=init_cfg)
        self.img_backbone = build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)

        # Far3D
        if img_roi_head is not None:
            self.img_roi_head = build_head(img_roi_head)
        else:
            self.img_roi_head = None

        if head is not None:
            self.head = build_head(head)
        else:
            self.head = None

        self.use_grid_mask = use_grid_mask
        if use_deformable_func:
            assert DAF_VALID, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )

        # Coop
        self.save_track_query = save_track_query
        self.save_track_query_file_root = save_track_query_file_root
        self.save_topK = save_topK
        self.save_threshold = save_threshold
        self.save_dn_for_coop = save_dn_for_coop
        if self.save_track_query:
            os.makedirs(self.save_track_query_file_root, exist_ok=True)
        # self.is_track_cooperation = is_track_cooperation

        # Freeze
        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck
        self.freeze_img_roi_head = freeze_img_roi_head
        if freeze_img_backbone:
            self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if freeze_img_neck:
            self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False
        if freeze_img_roi_head:
            self.img_roi_head.eval()
            for param in self.img_roi_head.parameters():
                param.requires_grad = False

    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_dense_depth=False, metas=None):
        bs = img.shape[0]
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)  # [B,N,C,H,W] -> [B*N,C,H,W]
        else:
            num_cams = 1
        if self.use_grid_mask:  # True
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
            # [[16, 256, 64, 176], [16, 512, 32, 88], [16, 1024, 16, 44], [16, 2048, 8, 22]]

        if self.img_neck is not None:
            feature_maps = list(self.img_neck(feature_maps))
            # [[16, 256, 64, 176], [16, 256, 32, 88], [16, 256, 16, 44], [16, 256, 8, 22]]

        for i, feat in enumerate(feature_maps):
            feature_maps[i] = torch.reshape(feat, (bs, num_cams) + feat.shape[1:])
            # [[4, 4, 256, 64, 176], [4, 4, 256, 32, 88], [4, 4, 256, 16, 44], [4, 4, 256, 8, 22]]

        if return_dense_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        else:
            depths = None
        if return_dense_depth:
            return feature_maps, depths
        return feature_maps

    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        feature_maps, dense_depths = self.extract_feat(
            img, return_dense_depth=True, metas=data
        )

        if self.img_roi_head is not None:  # Far3D
            assert "gt_bboxes_2d" in data
            outs_roi = self.img_roi_head(feature_maps, **data)
            bbox_dict = self.img_roi_head.get_bboxes(
                outs_roi, **data
            )  # {'bbox_list': BN x (Mi, 4), 'bbox2d_scores': (sum(Mi), 1), 'valid_indices': (B*N, sum(Hi*Wi), 1)}
            outs_roi.update(bbox_dict)
            data.update({"2d_proposals": outs_roi})

        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
            # [[4, 59840, 256], [4, 4, 2], [4, 4]]

        if self.head:
            model_outs = self.head(feature_maps, data)
            output = self.head.loss(model_outs, data)
        else:
            output = {}

        if dense_depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                dense_depths, data["gt_depth"]
            )
        if (
            self.img_roi_head is not None
            and "gt_bboxes_2d" in data
            and not self.freeze_img_roi_head
        ):
            loss2d_inputs = [
                data["gt_bboxes_2d"],
                data["gt_labels_2d"],
                data["centers_2d"],
                data["depths"],
                outs_roi,
                data["img_metas"],
            ]
            output.update(self.img_roi_head.loss(*loss2d_inputs))
        return output

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        feature_maps = self.extract_feat(img, metas=data)

        if self.img_roi_head is not None:  # Far3D
            outs_roi = self.img_roi_head(feature_maps, **data)
            bbox_dict = self.img_roi_head.get_bboxes(
                outs_roi, **data
            )  # {'bbox_list': BN x (Mi, 4), 'bbox2d_scores': (sum(Mi), 1), 'valid_indices': (B*N, sum(Hi*Wi), 1)}
            outs_roi.update(bbox_dict)
            data.update({"2d_proposals": outs_roi})

        if self.use_deformable_func:
            feature_maps = feature_maps_format(feature_maps)
            # [[4, 59840, 256], [4, 4, 2], [4, 4]]

        model_outs = self.head(feature_maps, data)
        results = self.head.post_process(model_outs, metas=data)

        # Save query for coop
        if self.save_track_query:
            for i in range(len(results)):
                if "cls_scores" in results[i]:
                    coop_score = results[i]["cls_scores"]
                else:
                    coop_score = results[i]["scores_3d"]

                if self.save_threshold is not None:
                    # Use threshold to save query
                    mask = coop_score >= self.save_threshold
                else:
                    # Use topK to save query
                    _, topK_idx = torch.topk(coop_score, self.save_topK)
                    mask = torch.zeros_like(coop_score, dtype=torch.bool)
                    mask[topK_idx] = True

                sample_idx = data["img_metas"][i]["sample_idx"]
                track_instances = {
                    "coop_instance_feature": results[i]["instance_feature"][mask],
                    "coop_instance_ids": results[i]["instance_ids"][mask],
                    "coop_anchor": results[i]["anchor"][mask],
                    "coop_label": results[i]["labels_3d"][mask],
                    "coop_score": results[i]["scores_3d"][mask],
                    "coop_cls_score": results[i]["cls_scores"][mask],
                    "coop_timestamp": data["img_metas"][i]["timestamp"],
                }

                if self.save_dn_for_coop:
                    track_instances.update(
                        {
                            "coop_dn_instance_feature": results[i][
                                "coop_dn_instance_feature"
                            ],
                            "coop_dn_anchor": results[i]["coop_dn_anchor"],
                            "coop_dn_id_target": results[i]["coop_dn_id_target"],
                            "coop_dn_cls_target": results[i]["coop_dn_cls_target"],
                            "coop_dn_reg_target": results[i]["coop_dn_reg_target"],
                        }
                    )
                mmcv.dump(
                    track_instances,
                    os.path.join(self.save_track_query_file_root, f"{sample_idx}.pkl"),
                )

        output = [{"img_bbox": result} for result in results]
        return output

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)
