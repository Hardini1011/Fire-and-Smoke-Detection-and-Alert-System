import streamlit as st
import cv2
import os
import uuid
from ultralytics import YOLO
import tempfile
import json
import subprocess
import numpy as np
from ultralytics.engine.results import Results
from _collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort
from stqdm import stqdm

from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()


def send_sms_alert(message_text="🚨 Fire or Smoke Detected!"):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM")
    to_number = os.getenv("TWILIO_TO")

    client = Client(account_sid, auth_token)
    message = client.messages.create(body=message_text, from_=from_number, to=to_number)
    print("SMS sent:", message.sid)


model = YOLO("best.pt")
COLORS = [
    (56, 56, 255),
    (151, 157, 255),
]


def result_to_json(result: Results, tracker=None):
    len_results = len(result.boxes)
    result_list_json = [
        {
            "class_id": int(result.boxes.cls[idx]),
            "class": result.names[int(result.boxes.cls[idx])],
            "confidence": float(result.boxes.conf[idx]),
            "bbox": {
                "x_min": int(result.boxes.data[idx][0]),
                "y_min": int(result.boxes.data[idx][1]),
                "x_max": int(result.boxes.data[idx][2]),
                "y_max": int(result.boxes.data[idx][3]),
            },
        }
        for idx in range(len_results)
    ]
    if tracker is not None:
        bbs = [
            (
                [
                    result_list_json[idx]["bbox"]["x_min"],
                    result_list_json[idx]["bbox"]["y_min"],
                    result_list_json[idx]["bbox"]["x_max"]
                    - result_list_json[idx]["bbox"]["x_min"],
                    result_list_json[idx]["bbox"]["y_max"]
                    - result_list_json[idx]["bbox"]["y_min"],
                ],
                result_list_json[idx]["confidence"],
                result_list_json[idx]["class"],
            )
            for idx in range(len_results)
        ]
        tracks = tracker.update_tracks(bbs, frame=result.orig_img)
        for idx in range(len(result_list_json)):
            track_idx = next(
                (
                    i
                    for i, track in enumerate(tracks)
                    if track.det_conf is not None
                    and np.isclose(track.det_conf, result_list_json[idx]["confidence"])
                ),
                -1,
            )
            if track_idx != -1:
                result_list_json[idx]["object_id"] = int(tracks[track_idx].track_id)
    return result_list_json


def view_result(result: Results, result_list_json, centers=None):
    image = result.plot(labels=False, line_width=2)
    for result in result_list_json:
        class_color = COLORS[result["class_id"] % len(COLORS)]
        text = (
            f"{result['class']} {result['object_id']}: {result['confidence']:.2f}"
            if "object_id" in result
            else f"{result['class']}: {result['confidence']:.2f}"
        )
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
        )
        cv2.rectangle(
            image,
            (result["bbox"]["x_min"], result["bbox"]["y_min"] - text_height - baseline),
            (result["bbox"]["x_min"] + text_width, result["bbox"]["y_min"]),
            class_color,
            -1,
        )
        cv2.putText(
            image,
            text,
            (result["bbox"]["x_min"], result["bbox"]["y_min"] - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        if "object_id" in result and centers is not None:
            centers[result["object_id"]].append(
                (
                    int((result["bbox"]["x_min"] + result["bbox"]["x_max"]) / 2),
                    int((result["bbox"]["y_min"] + result["bbox"]["y_max"]) / 2),
                )
            )
            for j in range(1, len(centers[result["object_id"]])):
                if (
                    centers[result["object_id"]][j - 1] is None
                    or centers[result["object_id"]][j] is None
                ):
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(
                    image,
                    centers[result["object_id"]][j - 1],
                    centers[result["object_id"]][j],
                    class_color,
                    thickness,
                )
    return image


def video_processing(video_file, model, tracker=None, centers=None):
    print("Video Processing...")
    results = model.predict(video_file)
    model_name = os.path.basename(model.ckpt_path).split(".")[0]
    output_folder = os.path.join("output_videos", video_file.split(".")[0])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video_file_name_out = os.path.join(
        output_folder, f"{video_file.split('.')[0]}_{model_name}_output.mp4"
    )
    if os.path.exists(video_file_name_out):
        os.remove(video_file_name_out)
    result_video_json_file = os.path.join(
        output_folder, f"{video_file.split('.')[0]}_{model_name}_output.json"
    )
    if os.path.exists(result_video_json_file):
        os.remove(result_video_json_file)
    json_file = open(result_video_json_file, "a")
    temp_file = "temp.mp4"
    video_writer = cv2.VideoWriter(
        temp_file,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (results[0].orig_img.shape[1], results[0].orig_img.shape[0]),
    )
    json_file.write("[\n")
    fire_or_smoke_detected = False
    for result in stqdm(results, desc=f"Processing video"):
        result_list_json = result_to_json(result, tracker=tracker)
        for detection in result_list_json:
            if detection["class"].lower() in ["fire", "smoke"]:
                fire_or_smoke_detected = True
        result_image = view_result(result, result_list_json, centers=centers)
        video_writer.write(result_image)
        json.dump(result_list_json, json_file, indent=2)
        json_file.write(",\n")
    if fire_or_smoke_detected:
        send_sms_alert()
    json_file.write("]")
    video_writer.release()
    subprocess.call(
        args=[
            "ffmpeg.exe",  # Path to your ffmpeg binary
            "-i",
            temp_file,
            "-c:v",
            "libx264",
            video_file_name_out,
        ]
    )
    os.remove(temp_file)
    return video_file_name_out, result_video_json_file


st.header("🚨 Fire and Smoke Detection and Alert System")
video_file = st.file_uploader("Upload a video", type=["mp4"])
process_video_button = st.button("Process Video")
if video_file is None and process_video_button:
    st.warning("Please upload a video file to be processed!")
if video_file is not None and process_video_button:
    print("Video Processing...")
    tracker = DeepSort(max_age=5)
    centers = [deque(maxlen=30) for _ in range(50)]
    open(video_file.name, "wb").write(video_file.read())
    video_file_out, result_video_json_file = video_processing(
        video_file.name, model, tracker=tracker, centers=centers
    )
    os.remove(video_file.name)
    video_bytes = open(video_file_out, "rb").read()
    st.video(video_bytes)
