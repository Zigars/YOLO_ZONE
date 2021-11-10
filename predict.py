'''
@Description:predict
@Author:Zigar
@Date:2021/11/03 14:12:30
'''
import argparse
import random
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
from utils.dataloader import LoadImages, LoadStreams

from utils.pred_utils import get_label, get_model, postprocess, preprocess, time_synchronized
from utils.general import increment_path
from utils.plots import plot_one_image


class Predictor(object):
    def __init__(
        self,
        input_size,
        device,
        num_classes,
        conf_thres,
        iou_thres,
        label_dir,
        weights_dir,
        version,
        source,
        view_img,
        save_txt,
        nosave,
        project,
        name,
        exist_ok
    ):
        super().__init__()
        # inference size (pixels)
        self.input_size = input_size

        # threshold in postprocess
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres


        # label/num_classes
        self.label = get_label(label_dir)
        self.num_classes = num_classes

        # inference device/dtype/model
        self.device = torch.device(device)
        self.model = get_model(self.num_classes, version, weights_dir, self.device)
        half = self.device != 'cpu'  # half precision only supported on CUDA
        if half:
            self.model.half()  # to FP16
        self.dtype = torch.float32 if not half else torch.float16

        # Dataloader
        # source: image / image_path / video / video_path / webcom
        # result save
        self.view_img, self.save_txt, self.nosave = view_img, save_txt,  nosave
        # get source type
        self.save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        self.webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        self.project, self.name, self.exist_ok = project, name, exist_ok
        self.save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Set Dataloader
        self.vid_path, self.vid_writer = None, None
        if self.webcam:
            # batch_size inference
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(source, img_size=input_size)
        else:
            self.dataset = LoadImages(source, img_size=input_size)

        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.label]\

    # predict for image / image_path / video / video_path / webcom
    def predict(self):
        # Run inference once for cuda
        if self.device != 'cpu':
            self.model(torch.zeros(1, 3, self.input_size, self.input_size).to(self.device).
                       type_as(next(self.model.parameters())))

        t0 = time.time()
        for path, image0, vid_cap in self.dataset:

            # get one image detection
            # data preprocess
            image, ratio = preprocess(image0, self.input_size, self.dtype, self.device)

            # model inference
            t1 = time_synchronized()
            outputs = self.model(image)
            t2 = time_synchronized()

            # data postprocess
            detections = postprocess(outputs, ratio, self.input_size, self.num_classes, self.conf_thres, self.iou_thres)
            t3 = time_synchronized()

            # path
            if self.webcam:  # batch_size >= 1
                p, s, image0, frame = path[0], '%g: ' % self.dataset.count, image0.copy(), self.dataset.count
            else:
                p, s, image0, frame = path, '', image0, getattr(self.dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(self.save_dir / p.name)  # img.jpg
            txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % image.shape[2:]  # print string

            if detections is not None:
                bboxes, scores, cls_ids = detections
                for c in np.unique(cls_ids):
                    n = (cls_ids == c).sum()
                    s += f"{n} {self.label[int(c)]}, "
                for bbox, score, cls_id in zip(bboxes, scores, cls_ids):
                    # write to file
                    if self.save_txt:
                        cls = self.label[cls_id]
                        line = list((cls, *bbox, score))  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write("{}".format(line) + '\n')

                # Print time (inference + NMS)
                print(f'{s}Done. ({t3 - t1:.3f}s)', f'(inference:{t2 - t1:.3f}s,',  f'nms:{t3 - t2:.3f}s)')

                image = plot_one_image(
                    image0, bboxes, scores, cls_ids, self.colors,
                    conf=self.conf_thres, class_names=self.label,
                )

                # Stream results
                if self.view_img and self.webcam:
                    cv2.imshow(str(p), image0)
                    cv2.waitKey(1)  # 1 millisecond

                # plot bboxes on image
                if self.save_img:
                    if self.dataset.mode == 'image':
                        cv2.imwrite(save_path, image)
                    else:  # 'video' or 'stream'
                        if self.vid_path != save_path:  # new video
                            self.vid_path = save_path
                            if isinstance(self.vid_writer, cv2.VideoWriter):
                                self.vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, image.shape[1], image.shape[0]
                                save_path += '.mp4'
                            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        self.vid_writer.write(image)

        if self.save_txt or self.save_img:
            print(f"Results saved to {self.save_dir}")
        print(f'Done. ({time.time() - t0:.3f}s)')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--device', default='cuda', help='cuda device,cuda or cpu')
    parser.add_argument('--num_classes', type=int, default=80, help='number of classes')
    parser.add_argument('--weights-dir', nargs='+', type=str, default='weights/yolox_s.pth', help='model.pth path(s)')
    parser.add_argument('--version', type=str, default='s', help='version of models')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--label-dir', type=str, default='labels/coco_label.txt', help='label names')
    parser.add_argument('--view-img', type=bool, default=True, help='display results')
    parser.add_argument('--save-txt', type=bool, default=True, help='save results to *.txt')
    parser.add_argument('--nosave', type=bool, default=False, help='do not save images/videos')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')

    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')


    opt = parser.parse_args()
    print(opt)


    with torch.no_grad():
        Predictor(
            opt.input_size,
            opt.device,
            opt.num_classes,
            opt.conf_thres,
            opt.iou_thres,
            opt.label_dir,
            opt.weights_dir,
            opt.version,
            opt.source,
            opt.view_img,
            opt.save_txt,
            opt.nosave,
            opt.project,
            opt.name,
            opt.exist_ok
        ).predict()

