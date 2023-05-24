import os
import cv2
import pafy
import sys
import time
from queue import Queue
from threading import Thread
import torch

from VideoESRGAN import esrgan
# from VideoRIFE import rife_model
from GoogleFiLM import google_film_model


def frame_interpolation(vfiQ: Queue, resultQ: Queue):
    global last_frame_id, video_processed
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    film_model = google_film_model()
    frame1 = frame2 = None

    frame_id = 0
    try:
        while last_frame_id == -1 or frame_id < last_frame_id:
            # print(f"last_frame_id: {last_frame_id}, frame_id: {frame_id}")
            frame2 = vfiQ.get()
            frame_id += 1

            if frame1 is None:
                frame1 = frame2
                resultQ.put(frame1)
                cv2.imshow('interpolated', frame1)
                continue
            else:
                print(f"interpolating_frame no. {frame_id-1} and {frame_id}.. wait for some time...")
                # print(f"frame1.shape: {frame1.shape}, frame2.shape: {frame2.shape}")
                start_time = time.time()
                intm_frame = film_model.interpolate_frame(frame1, frame2)
                print(f"interpolation time: {time.time() - start_time}")
                frame1 = frame2
                # print(f"interpolated_frame.shape: {intm_frame.shape}, type: {type(intm_frame)}")
                # print(f"frame1.shape: {frame1.shape}, type: {type(frame1)}")
                resultQ.put(intm_frame)
                resultQ.put(frame2)
                cv2.imshow('interpolated', intm_frame)
                cv2.waitKey(1)
                cv2.imshow('interpolated', frame2)

            
            cv2.waitKey(1)
            # print(f"last_frame_id == -1 or frame_id <= last_frame_id: {last_frame_id == -1 or frame_id <= last_frame_id}")

        print("frame_interpolation completed")

    except Exception as e:
        print('\nframe_iterpolation stopped due to:', e)

    cv2.destroyWindow('interpolated')
    video_processed = True


def frame_enhance(enhanceQ: Queue, vfiQ: Queue):
    global last_frame_id
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    esrgan_model = esrgan(gpu_id=0)
    frame_id = 0
    try:
        # while ret or not enhanceQ.not_empty:
        while last_frame_id == -1 or frame_id < last_frame_id:
            frame = enhanceQ.get()
            frame_id += 1
            output_frame = esrgan_model.enhance(frame)
            height, width, _ = frame.shape
            # print(f"enhanced_frame.shape: {output_frame.shape}, type: {type(output_frame)}")
            output_frame = cv2.resize(output_frame, (int(width/height *480), 480))
            vfiQ.put(output_frame)
            cv2.imshow('enhanced', output_frame)
            cv2.waitKey(1)
    except Exception as e:
        print('\nframe_enhance stopped due to:', e)
    del esrgan_model
    cv2.destroyWindow('enhanced')


def video_output(resultQ: Queue):
    global video_processed, fps

    i = 0
    output_path = f'output/output-{i}'
    # Create a directory to store the frames
    while os.path.exists(output_path):
        i += 1
        output_path = f'output/output-{i}'

    os.makedirs(output_path)


    video_out_writer = None
    
    try:
        frame_no = 0
        print("Writing frames to video...")
        print(f"resultQ.empty(): {resultQ.empty()}, video_processed: {video_processed}")
        while video_processed is False or not resultQ.empty():
            frame = resultQ.get()
            frame_no += 1
            # print(f"frame_no: {frame_no}")

            if video_out_writer is None:
                video_out_writer = cv2.VideoWriter(f'{output_path}/enhanced_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))

            cv2.imwrite(f'{output_path}/frame{frame_no}.jpg', frame)
            video_out_writer.write(frame)

    except Exception as e:
        print('\nvideo_output stopped due to:', e)

    video_out_writer.release()
    print("Video Enhancement Completed..!!")


if __name__ == '__main__':
    ret = True
    video_urls = [
        "https://www.youtube.com/watch?v=V9DWKbalbWQ",
        "https://www.youtube.com/watch?v=QDX-1M5Nj7s",
        "https://www.youtube.com/watch?v=IPfo1k2JyIg",
        "https://www.youtube.com/watch?v=a2uKphzsjMo",
        "https://www.youtube.com/watch?v=I1J2Z_Fgado",
    ]

    url = video_urls[2]
    video = pafy.new(url)
    res = []
    for s in video.streams:
        # print(s.resolution, s.extension, s.get_filesize())
        res.append(s.resolution)
        url = s.url
        break
    print(sorted(res), url)

    print(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # cam = cv2.VideoCapture("videoplayback.3gp")
    cam = cv2.VideoCapture(url)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print(f'Frame Rate: {fps}fps')
    print(f'Frame Size: {cam.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cam.get(cv2.CAP_PROP_FRAME_HEIGHT)}')

    vfiQ = Queue()
    enhanceQ = Queue()
    resultQ = Queue()

    p1 = Thread(target=frame_interpolation, args=(vfiQ, resultQ,), daemon=True)
    p2 = Thread(target=frame_enhance, args=(enhanceQ, vfiQ), daemon=True)
    p3 = Thread(target=video_output, args=(resultQ,), daemon=True)

    p1.start()
    p2.start()
    p3.start()

    video_processed = False

    frame_id = 0
    last_frame_id = -1
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            frame_id += 1
            # vfiQ.put(frame)
            enhanceQ.put(frame)
            cv2.imshow('original', frame)
            cv2.waitKey(1)
            print(frame_id, end=" ")
            # time.sleep(1/fps)
        last_frame_id = frame_id
    except:
        ret = False
        cv2.destroyWindow('original')
        print("Feed Ended or Error occured")

    
    print(f"last_frame_id: {last_frame_id}")

    cv2.destroyWindow('original')
    

    p1.join()
    p2.join()
    p3.join()

    sys.exit(0)
