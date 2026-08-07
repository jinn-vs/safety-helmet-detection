import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

person_model = YOLO("yolov8n.pt")
helmet_model = YOLO("best.pt")

def get_iou_top(person_box, head_box):
    px1, py1, px2, py2 = person_box
    p_mid_y = (py1 + py2) / 2
    hx1, hy1, hx2, hy2 = head_box
    ix1 = max(px1, hx1)
    iy1 = max(py1, hy1)
    ix2 = min(px2, hx2)
    iy2 = min(p_mid_y, hy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    head_area = max((hx2 - hx1) * (hy2 - hy1), 1e-6)
    return intersection / head_area

def smart_detect(input_image):
    if input_image is None:
        return None, ""
    img = input_image.copy()
    person_results = person_model.predict(img, conf=0.50, verbose=False)
    persons = []
    for box in person_results[0].boxes:
        if int(box.cls[0]) == 0:
            coords = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            persons.append({"coords": coords, "conf": conf, "status": "no-helmet"})
    helmet_results = helmet_model.predict(img, conf=0.45, verbose=False)
    helmets, no_helmets = [], []
    for box in helmet_results[0].boxes:
        cls_id = int(box.cls[0])
        coords = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        if cls_id == 0:
            helmets.append({"coords": coords, "conf": conf})
        elif cls_id == 1:
            no_helmets.append({"coords": coords, "conf": conf})
    for person in persons:
        px1, py1, px2, py2 = person["coords"]
        p_mid_y = (py1 + py2) / 2
        best_overlap = 0.2
        for helmet in helmets:
            hx1, hy1, hx2, hy2 = helmet["coords"]
            if hy1 < p_mid_y and hx1 < px2 and hx2 > px1:
                overlap = get_iou_top([px1, py1, px2, p_mid_y], [hx1, hy1, hx2, hy2])
                if overlap > best_overlap:
                    best_overlap = overlap
                    person["status"] = "helmet"
        for nh in no_helmets:
            nx1, ny1, nx2, ny2 = nh["coords"]
            if ny1 < p_mid_y and nx1 < px2 and nx2 > px1:
                overlap = get_iou_top([px1, py1, px2, p_mid_y], [nx1, ny1, nx2, ny2])
                if overlap > best_overlap:
                    best_overlap = overlap
                    person["status"] = "no-helmet"
    for person in persons:
        px1, py1, px2, py2 = [int(c) for c in person["coords"]]
        if person["status"] == "helmet":
            color = (0, 200, 0)
            label = "SAFE"
        else:
            color = (0, 0, 230)
            label = "UNSAFE"
        cv2.rectangle(img, (px1, py1), (px2, py2), color, 3)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        bar_h = text_size[1] + 16
        cv2.rectangle(img, (px1, py1 - bar_h), (px1 + text_size[0] + 10, py1), color, -1)
        cv2.putText(img, label, (px1 + 5, py1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        conf_text = f"{person['conf']:.0%}"
        cv2.putText(img, conf_text, (px1 + 5, py2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    helmet_count = sum(1 for p in persons if p["status"] == "helmet")
    no_helmet_count = sum(1 for p in persons if p["status"] == "no-helmet")
    summary = "Detection Results\n\n"
    summary += f"Total Persons Detected: {len(persons)}\n"
    summary += f"With Helmet: {helmet_count}\n"
    summary += f"Without Helmet: {no_helmet_count}\n\n"
    if len(persons) == 0:
        summary += "No person detected in the image."
    elif no_helmet_count > 0:
        summary += f"Alert: {no_helmet_count} person(s) detected without helmet."
    else:
        summary += "All detected persons are wearing helmets."
    return img_rgb, summary

with gr.Blocks(title="Helmet Detection System", theme=gr.themes.Soft()) as demo:
    gr.HTML("<div style='text-align:center;padding:20px 0 10px 0;'>"
           "<p style='font-size:28px;font-weight:700;'>Helmet Detection System</p>"
           "<p style='font-size:14px;color:#666;'>AI-ML Project - Safety Helmet Detection Using YOLOv8</p>"
           "<p style='font-size:12px;color:#888;'>Developed by Muhammad Ali Qamer</p></div>")
    with gr.Tab("Image Upload"):
        gr.Markdown("Upload an image to run helmet detection.")
        with gr.Row():
            img_input = gr.Image(type="numpy", label="Upload Image")
            with gr.Column():
                img_output = gr.Image(label="Detection Result")
                txt_output = gr.Textbox(label="Summary", lines=10)
        img_btn = gr.Button("Detect", variant="primary")
        img_btn.click(fn=smart_detect, inputs=img_input, outputs=[img_output, txt_output])
    gr.HTML("<p style='text-align:center;font-size:11px;color:#999;padding:10px;'>This system is intended for demonstration purposes only. Detection accuracy may vary depending on image quality, lighting conditions, and camera angle.</p>")

if __name__ == "__main__":
    demo.launch()