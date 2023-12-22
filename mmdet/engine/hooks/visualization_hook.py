# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os
import json
import warnings
from typing import Optional, Sequence
import numpy as np

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmdet.datasets.samplers import TrackImgSampler
from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample, TrackDataSample

class ClassCounter:
    def __init__(self, classes):
        self.classes = classes
        self.gt_counts = {class_label: np.array([]) for class_label in classes}
        self.pred_counts = {class_label: np.array([]) for class_label in classes}

    def add_ground_truth(self, class_label, count):
        self.gt_counts[class_label] = np.append(self.gt_counts[class_label], count)

    def add_prediction(self, class_label, count):
        self.pred_counts[class_label] = np.append(self.pred_counts[class_label], count)

    def mean_absolute_error(self):
        mae_per_class = {
            class_label: np.mean(np.abs(self.pred_counts[class_label] - self.gt_counts[class_label]))
            for class_label in self.classes
        }
        return mae_per_class

    def root_mean_squared_error(self):
        rmse_per_class = {
            class_label: np.sqrt(np.mean((self.pred_counts[class_label] - self.gt_counts[class_label])**2))
            for class_label in self.classes
        }
        return rmse_per_class

@HOOKS.register_module()
class DetVisualizationHook(Hook):
    """Detection Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 backend_args: dict = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args
        self.draw = draw
        self.test_out_dir = test_out_dir
        self._test_index = 0

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[DetDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        img_path = outputs[0].img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'val_img',
                img,
                data_sample=outputs[0],
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                step=total_curr_iter)
        

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[DetDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        classes = ('Abnormal', 'Flower', 'Ripe', 'Underripe', 'Unripe')

        class_counter = ClassCounter(classes)

        for data_sample in outputs:
            self._test_index += 1

            img_path = data_sample.img_path
            img_bytes = get(img_path, backend_args=self.backend_args)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

            out_file = None
            if self.test_out_dir is not None:
                out_file = osp.basename(img_path)
                out_file = osp.join(self.test_out_dir, out_file)

            self._visualizer.add_datasample(
                osp.basename(img_path) if self.show else 'test_img',
                img,
                data_sample=data_sample,
                show=self.show,
                wait_time=self.wait_time,
                pred_score_thr=self.score_thr,
                out_file=out_file,
                step=self._test_index)
            
            # classes = self.dataset_meta.get('classes', None)
            
            output_filename = f"{out_file}.json"
            classes = ('Abnormal', 'Flower', 'Ripe', 'Underripe', 'Unripe')

            # Helper function to process instances
            def process_instances(instances, score_thr):
                if instances is not None and hasattr(instances, 'bboxes'):
                    filtered_instances = instances[instances.scores > score_thr] if hasattr(instances, 'scores') else instances
                    total_box = filtered_instances.bboxes
                    box_positions = total_box[:, :2]
                    labels = filtered_instances.labels
                    return total_box, box_positions, labels
                else:
                    return 0, 0, None

            # Process ground truth instances
            total_gt_box, gt_box_positions, gt_labels = process_instances(getattr(data_sample, 'gt_instances', None), self.score_thr)

            # Process predicted instances
            total_pred_box, pred_box_positions, pred_labels = process_instances(getattr(data_sample, 'pred_instances', None), self.score_thr)

            # Initialize dictionaries using dict.fromkeys
            gt_class_quantity_dict = dict.fromkeys(classes, 0)
            pred_class_quantity_dict = dict.fromkeys(classes, 0)

            output_data = {
                "Score Thr": self.score_thr,
                "Ground Truth": {},
                "Prediction": {}
            }

            for data, class_quantity_dict in zip([(total_gt_box, gt_labels), (total_pred_box, pred_labels)],
                                    [gt_class_quantity_dict, pred_class_quantity_dict]):
                for pos, label in zip(*data):
                    if 0 <= label < len(classes):
                        class_name = classes[label]
                        class_quantity_dict[class_name] += 1

                # Update the output_data dictionary
                section_name = "Ground Truth" if class_quantity_dict is gt_class_quantity_dict else "Prediction"
                output_data[section_name] = {class_label: class_quantity_dict[class_label] for class_label in classes}

            # Write the data to a JSON file
            with open(output_filename, 'w') as f:
                json.dump(output_data, f, indent=2)

            # with open(output_filename, 'w') as f:
            #     f.write(f"Score Thr: {self.score_thr}\n")

            #     # Combine the loops for ground truth and prediction
            #     for data, class_quantity_dict in zip([(total_gt_box, gt_labels), (total_pred_box, pred_labels)],
            #                                         [gt_class_quantity_dict, pred_class_quantity_dict]):
            #         for pos, label in zip(*data):
            #             if 0 <= label < len(classes):
            #                 class_name = classes[label]
            #                 class_quantity_dict[class_name] += 1

            #         # Write the results for ground truth and prediction
            #         section_name = "Ground Truth" if class_quantity_dict is gt_class_quantity_dict else "Prediction"
            #         f.write(f"\n{section_name}\n")
            #         # f.write(f"\n{section_name}\n")
                    
            #         for class_label in classes:
            #             if class_quantity_dict is gt_class_quantity_dict:
            #                 class_counter.add_ground_truth(class_label, class_quantity_dict[class_label])
            #             else:
            #                 class_counter.add_prediction(class_label, class_quantity_dict[class_label])
            #             f.write(f"Class {class_label}: {class_quantity_dict[class_label]}\n")


        # output_all_res = f"{self.test_out_dir}/res_all"
        # os.makedirs(output_all_res)
        # output_filename_res = f"{output_all_res}/res.txt"

        # with open(output_filename_res, 'w') as f:
        #     f.write("MAE:\n")
        #     f.write(str(class_counter.mean_absolute_error()))
        #     f.write("\nRMSE:\n")
        #     f.write(str(class_counter.root_mean_squared_error()))
        


@HOOKS.register_module()
class TrackVisualizationHook(Hook):
    """Tracking Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        frame_interval (int): The interval of visualization. Defaults to 30.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        backend_args (dict): Arguments to instantiate a file client.
            Defaults to ``None``.
    """

    def __init__(self,
                 draw: bool = False,
                 frame_interval: int = 30,
                 score_thr: float = 0.3,
                 show: bool = False,
                 wait_time: float = 0.,
                 test_out_dir: Optional[str] = None,
                 backend_args: dict = None) -> None:
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.frame_interval = frame_interval
        self.score_thr = score_thr
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args
        self.draw = draw
        self.test_out_dir = test_out_dir
        self.image_idx = 0

    def after_val_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                       outputs: Sequence[TrackDataSample]) -> None:
        """Run after every ``self.interval`` validation iteration.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`TrackDataSample`]): Outputs from model.
        """
        if self.draw is False:
            return

        assert len(outputs) == 1,\
            'only batch_size=1 is supported while validating.'

        sampler = runner.val_dataloader.sampler
        if isinstance(sampler, TrackImgSampler):
            if self.every_n_inner_iters(batch_idx, self.frame_interval):
                total_curr_iter = runner.iter + batch_idx
                track_data_sample = outputs[0]
                self.visualize_single_image(track_data_sample[0],
                                            total_curr_iter)
        else:
            # video visualization DefaultSampler
            if self.every_n_inner_iters(batch_idx, 1):
                track_data_sample = outputs[0]
                video_length = len(track_data_sample)

                for frame_id in range(video_length):
                    if frame_id % self.frame_interval == 0:
                        total_curr_iter = runner.iter + self.image_idx + \
                                          frame_id
                        img_data_sample = track_data_sample[frame_id]
                        self.visualize_single_image(img_data_sample,
                                                    total_curr_iter)
                self.image_idx = self.image_idx + video_length

    def after_test_iter(self, runner: Runner, batch_idx: int, data_batch: dict,
                        outputs: Sequence[TrackDataSample]) -> None:
        """Run after every testing iteration.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`TrackDataSample`]): Outputs from model.
        """
        if self.draw is False:
            return

        assert len(outputs) == 1, \
            'only batch_size=1 is supported while testing.'

        if self.test_out_dir is not None:
            self.test_out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.test_out_dir)
            mkdir_or_exist(self.test_out_dir)

        sampler = runner.test_dataloader.sampler
        if isinstance(sampler, TrackImgSampler):
            if self.every_n_inner_iters(batch_idx, self.frame_interval):
                track_data_sample = outputs[0]
                self.visualize_single_image(track_data_sample[0], batch_idx)
        else:
            # video visualization DefaultSampler
            if self.every_n_inner_iters(batch_idx, 1):
                track_data_sample = outputs[0]
                video_length = len(track_data_sample)

                for frame_id in range(video_length):
                    if frame_id % self.frame_interval == 0:
                        img_data_sample = track_data_sample[frame_id]
                        self.visualize_single_image(img_data_sample,
                                                    self.image_idx + frame_id)
                self.image_idx = self.image_idx + video_length

    def visualize_single_image(self, img_data_sample: DetDataSample,
                               step: int) -> None:
        """
        Args:
            img_data_sample (DetDataSample): single image output.
            step (int): The index of the current image.
        """
        img_path = img_data_sample.img_path
        img_bytes = get(img_path, backend_args=self.backend_args)
        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')

        out_file = None
        if self.test_out_dir is not None:
            video_name = img_path.split('/')[-3]
            mkdir_or_exist(osp.join(self.test_out_dir, video_name))
            out_file = osp.join(self.test_out_dir, video_name,
                                osp.basename(img_path))

        self._visualizer.add_datasample(
            osp.basename(img_path) if self.show else 'test_img',
            img,
            data_sample=img_data_sample,
            show=self.show,
            wait_time=self.wait_time,
            pred_score_thr=self.score_thr,
            out_file=out_file,
            step=step)
