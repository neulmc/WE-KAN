from mmengine.hooks import Hook
from mmrotate.structures.bbox import rbbox_overlaps
import pickle
from mmengine.logging import print_log
import os.path as osp
import os

class SaveDetectionHook(Hook):
    """保存检测结果的钩子"""

    def __init__(self, save_dir='detection_results'):
        self.save_dir = save_dir
        self.results = []

    def after_test_iter(self, runner, batch_idx, data_batch, outputs):
        """每次测试迭代后调用"""

        import torch

        if not isinstance(outputs, list):
            return

        for i in range(len(outputs)):
            pred = outputs[i]

            if i < len(data_batch['data_samples']):
                data_sample = data_batch['data_samples'][i]
                gt_bboxes = data_sample.gt_instances.bboxes.tensor  # [n, 5]
                gt_labels = data_sample.gt_instances.labels
                img_name = data_sample.img_path if hasattr(data_sample, 'img_path') else f'img_{i}'
            else:
                continue

            pred_instances = pred.pred_instances

            if pred_instances is None or not hasattr(pred_instances, 'bboxes'):
                continue

            pred_bbox = pred_instances.bboxes  # [m, 5]
            pred_scores = pred_instances.scores
            pred_labels = pred_instances.labels

            for j in range(len(pred_bbox)):
                # 单个预测框，需要添加batch维度: [1, 1, 5]
                if pred_scores[j] < 0.15:  # 添加这行
                    continue  # 跳过这个预测
                single_pred_bbox = pred_bbox[j:j + 1]#.unsqueeze(0)  # [1, 1, 5]

                # 真值框也需要添加batch维度: [1, n, 5]
                gt_bboxes_batch = gt_bboxes#.unsqueeze(0)  # [1, n, 5]

                ious_tensor = rbbox_overlaps(
                    single_pred_bbox.cpu(),
                    gt_bboxes_batch.cpu(),
                    mode='iou',
                    is_aligned=False
                )  # 根据文档，输出应该是 [1, n]

                # 去掉batch维度
                # ious = ious_tensor[0] if ious_tensor.dim() > 1 else ious_tensor
                ious_np = ious_tensor.cpu().numpy()

                result = {
                    'batch_idx': batch_idx,
                    'image_idx': i,
                    'img_name': img_name,  # 添加这行
                    'pred_bbox': pred_bbox[j].cpu().numpy().tolist(),
                    'pred_score': float(pred_scores[j].cpu().numpy()),
                    'pred_label': int(pred_labels[j].cpu().numpy()),
                    'gt_bboxes': gt_bboxes.cpu().numpy().tolist(),
                    'gt_labels': gt_labels.cpu().numpy().tolist(),
                    'ious': ious_np.tolist() if hasattr(ious_np, 'tolist') else ious_np
                }
                self.results.append(result)


    def after_test(self, runner):
        """测试结束后调用"""
        save_dir = osp.join(runner.work_dir, self.save_dir)
        os.makedirs(save_dir, exist_ok=True)

        # 保存为pickle文件
        with open(osp.join(save_dir, 'detection_results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)

        # 也保存为JSON（可选）
        import json
        with open(osp.join(save_dir, 'detection_results.json'), 'w') as f:
            # 需要将numpy数组转换为列表
            json_results = []
            for res in self.results:
                json_res = {k: (v.tolist() if hasattr(v, 'tolist') else v)
                            for k, v in res.items()}
                json_results.append(json_res)
            json.dump(json_results, f, indent=2)

        print_log(f'Detection results saved to {save_dir}', logger='current')