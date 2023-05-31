import os
import cv2
import time
import subprocess
from queue import Queue
from threading import Thread

from .VideoESRGAN import esrgan
from .GoogleFiLM import google_film_model

from .drive import upload_file


def frame_interpolation(vfiQ: Queue, resultQ: Queue):
    global total_frames
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    film_model = google_film_model()
    frame1 = frame2 = None

    frame_id = 0
    try:
        while frame_id < total_frames:
            frame2 = vfiQ.get()
            frame_id += 1

            if frame1 is None:
                frame1 = frame2
                resultQ.put(frame1)
                # cv2.imshow('interpolated', frame1)
                continue
            else:
                start_time = time.time()
                intm_frame = film_model.interpolate_frame(frame1, frame2)
                print(f"Interpolated frame no. {frame_id-1} and {frame_id} in {time.time() - start_time} seconds")
                
                frame1 = frame2
                resultQ.put(intm_frame)
                resultQ.put(frame2)
                # cv2.imshow('interpolated', intm_frame)
                # cv2.waitKey(1)
                # cv2.imshow('interpolated', frame2)

            # cv2.waitKey(1)

        print("Frame Interpolation Completed")

    except Exception as e:
        print('\nframe_iterpolation stopped due to:', e)

    # cv2.destroyWindow('interpolated')


def frame_enhance(enhanceQ: Queue, vfiQ: Queue):
    global total_frames
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    esrgan_model = esrgan(gpu_id=0)
    frame_id = 0
    try:
        # while ret or not enhanceQ.not_empty:
        while frame_id < ((total_frames * 2) - 1):
            frame = enhanceQ.get()
            frame_id += 1

            start_time = time.time()
            output_frame = esrgan_model.enhance(frame)
            print(f"Enhanced frame no. {frame_id} in {time.time() - start_time} seconds")

            height, width, _ = frame.shape
            max_height = 480
            output_frame = cv2.resize(output_frame, (int(width/height *max_height), max_height))
            vfiQ.put(output_frame)
            # cv2.imshow('enhanced', output_frame)
            # cv2.waitKey(1)

        print("Frame Enhancement Completed")
    except Exception as e:
        print('\nframe_enhance stopped due to:', e)
    del esrgan_model
    # cv2.destroyWindow('enhanced')


def video_output(resultQ: Queue, request_id: str):
    global fps, enhanced_video_url

    i = 0
    output_path = f'enhance/output/output-{i}'
    # Create a directory to store the frames
    while os.path.exists(output_path):
        i += 1
        output_path = f'enhance/output/output-{i}'

    os.makedirs(output_path)
    enhance_fname = f'{output_path}/enhanced.mp4'
    filename = f'{output_path}/{request_id}.mp4'

    audio_path = f'enhance/.temp/audio/{request_id}.m4a'

    video_out_writer = None

    try:
        frame_no = 0
        print("Writing frames to video...")
        while frame_no < ((total_frames * 2) - 1) or not resultQ.empty():
            frame = resultQ.get()
            frame_no += 1
            # print(f"frame_no: {frame_no}")

            if video_out_writer is None:
                video_out_writer = cv2.VideoWriter(enhance_fname, cv2.VideoWriter_fourcc(*'mp4v'), fps*2, (frame.shape[1], frame.shape[0]))

            # cv2.imwrite(f'{output_path}/frame{frame_no}.jpg', frame)
            video_out_writer.write(frame)

    except Exception as e:
        print('\nvideo_output stopped due to:', e)

    video_out_writer.release()

    # merge the audio and video
    print("Merging audio and video...")
    subprocess.run(
        f'ffmpeg -y -i {enhance_fname} -i {audio_path} -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 -shortest {filename}', shell=True)

    enhanced_video_url = upload_file.upload_file(filename)
    print("Video Enhancement Completed..!!")


def main(url: str, request_id: str):
    global fps, total_frames, enhanced_video_url
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)

    try:
        cap = cv2.VideoCapture(url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                      cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f'Frame Rate: {fps}fps')
        print(f'Frame Size: {frame_size}')
        print(f"Total no. of frames: {total_frames}")

        video_duration = total_frames / fps
        print(f"Video Duration: {video_duration} seconds")
    except Exception as e:
        print(f"Exception: {e}")
        return None, "failed", "Video enhancement failed due to invalid url"
    
    # Create a directory to store the audio
    audio_path = f'enhance/.temp/{request_id}/audio'
    os.makedirs(audio_path, exist_ok=True)

    # get the audio stream using ffmpeg
    subprocess.Popen(f'ffmpeg -y -i {url} -vn -acodec copy {audio_path}/{request_id}.m4a', shell=True)

    enhanced_video_url = None

    frame_id = 0

    vfiQ = Queue()
    enhanceQ = Queue()
    resultQ = Queue()

    p1 = Thread(target=frame_enhance, args=(enhanceQ, vfiQ), daemon=True)
    p2 = Thread(target=frame_interpolation, args=(vfiQ, resultQ,), daemon=True)
    p3 = Thread(target=video_output, args=(resultQ, request_id), daemon=True)

    p1.start()
    p2.start()
    p3.start()

    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # vfiQ.put(frame)
            enhanceQ.put(frame)
            frame_id += 1

            # cv2.imshow('original', frame)
            # cv2.waitKey(1)
            print(frame_id, end=" ")

    except:
        # cv2.destroyWindow('original')
        print("Feed Ended or Error occured")

    # cv2.destroyWindow('original')

    p1.join()
    p2.join()
    p3.join()

    print(
        f"Video Duration: {video_duration} seconds, Time taken to enhance: {time.time() - start_time} seconds")

    return enhanced_video_url, "success", "Video enhanced successfully"
