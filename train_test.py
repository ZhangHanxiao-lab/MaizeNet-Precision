from ultralytics import YOLO
from ultralytics.cfg import get_cfg

def yolov8_MLSKAP(cfg):
    cfg = get_cfg(cfg)
    model = YOLO(cfg.model)
    model.train(**vars(cfg))

def test():
    model = YOLO('./logs/data1/yolov8s_M_H_P/weights/best.pt')
    # Validate the model
    results = model.val(
        data='./data/corn.yaml',
        split='test',
        imgsz=1024,
        device=0,
        batch=8,
        save_json=True,
        project='logs/test',
        name='yolov8s_M_H_P'
    )
    print(f"mAP50: {results.box.map50}")
    print(f"mAP50-95: {results.box.map}")
    print(f"Precision: {results.box.p}")
    print(f"Recall: {results.box.r}")

def main():
    # ------------------------test------------------------------
    # test()
    # ------------------------train------------------------------
    cfg = './config/data1/yolov8_config_M_H_P.yaml'
    yolov8_MLSKAP(cfg)


if __name__ == '__main__':
   main()