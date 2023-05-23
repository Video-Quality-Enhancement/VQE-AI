import os
import cv2
# import pafy
import sys
import time
from queue import Queue
from threading import Thread
import torch
import gc

from VideoESRGAN import esrgan
# from VideoRIFE import rife_model
from GoogleFiLM import GoogleFiLM


def frame_interpolation(vfiQ: Queue, resultQ: Queue):
    global ret
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    google_film = GoogleFiLM()
    # time.sleep(60)
    frame1 = frame2 = None

    try:
        frame_no = 0
        while True:
            frame2 = vfiQ.get()
            frame_no += 1

            if frame1 is None:
                frame1 = frame2
                resultQ.put(frame1)
                continue
            else:
                print(f"interpolating_frame no. {frame_no} and {frame_no+1}.. wait for some time...")
                # print(f"frame1.shape: {frame1.shape}, frame2.shape: {frame2.shape}")
                intm_frame = google_film.interpolate_frame(frame1, frame2)
                resultQ.put(intm_frame)
                resultQ.put(frame2)
    except Exception as e:
        print('\nframe_iterpolation stopped due to:', e)


def frame_enhance(enhanceQ: Queue, vfiQ: Queue):
    global ret
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    esrgan_model = esrgan(gpu_id=0)

    try:
        # while ret or not enhanceQ.not_empty:
        while True:
            frame = enhanceQ.get()
            output_frame = esrgan_model.enhance(frame)
            vfiQ.put(output_frame)
            cv2.imshow('enhanced', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print('\nframe_enhance stopped due to:', e)


def video_output(resultQ: Queue):
    global ret

    output_path = 'output'

    # Create a directory to store the frames
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    try:
        frame_no = 0
        while True:
            frame = resultQ.get()
            frame_no += 1
            cv2.imwrite(f'{output_path}/frame{frame_no}.jpg', frame)
    except Exception as e:
        print('\nvideo_output stopped due to:', e)


if __name__ == '__main__':
    ret = True
    # video_urls = [
    #     "https://www.youtube.com/watch?v=V9DWKbalbWQ",
    #     "https://www.youtube.com/watch?v=QDX-1M5Nj7s",
    #     "https://www.youtube.com/watch?v=IPfo1k2JyIg",
    #     "https://www.youtube.com/watch?v=a2uKphzsjMo",
    #     "https://www.youtube.com/watch?v=I1J2Z_Fgado",
    # ]

    # url = video_urls[3]
    # video = pafy.new(url)
    # res = []
    # for s in video.streams:
    #     # print(s.resolution, s.extension, s.get_filesize())
    #     res.append(s.resolution)
    #     url = s.url
    #     break

    print(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    # print(sorted(res), url)
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)

    cam = cv2.VideoCapture("videoplayback.3gp")
    # cam = cv2.VideoCapture(url)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print(f'Frame Rate: {fps}fps')

    vfiQ = Queue()
    enhanceQ = Queue()
    resultQ = Queue()

    p1 = Thread(target=frame_interpolation, args=(vfiQ, enhanceQ,), daemon=True)
    p2 = Thread(target=frame_enhance, args=(enhanceQ, resultQ), daemon=True)
    p3 = Thread(target=video_output, args=(resultQ,), daemon=True)

    p1.start()
    p2.start()
    p3.start()

    frame_id = 0
    try:
        while True:
            ret, frame = cam.read()
            frame_id += 1
            vfiQ.put(frame)
            # enhanceQ.put(frame)
            cv2.imshow('test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print(frame_id, end=" ")
            time.sleep(1/fps)
    except:
        ret = False
        cv2.destroyAllWindows()
        print("Feed Ended or Error occured")
    

    p1.join()
    p2.join()
    p3.join()

    sys.exit(0)
