import os
import cv2
import time
import subprocess
import ffmpeg
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

from .VideoESRGAN import esrgan
from .GoogleFiLM import google_film_model, google_film_onnx

from .drive import upload_file


def frame_interpolation(vfiQ: Queue, enhanceQ: Queue):
    global total_frames
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    film_model = google_film_onnx()
    frame1 = frame2 = None

    frame_id = 0
    try:
        while frame_id < total_frames:
            # print(f"total_frames: {total_frames}, frame_id: {frame_id}")
            frame2 = vfiQ.get()
            frame_id += 1

            if frame1 is None:
                frame1 = frame2
                enhanceQ.put(frame1)
                # cv2.imshow('interpolated', frame1)
                continue
            else:
                start_time = time.time()
                intm_frame = film_model.interpolate_frame(frame1, frame2)
                print(f"Interpolated frame no. {frame_id-1} and {frame_id} in {time.time() - start_time} seconds")
                frame1 = frame2
                # print(f"interpolated_frame.shape: {intm_frame.shape}, type: {type(intm_frame)}")
                # print(f"frame1.shape: {frame1.shape}, type: {type(frame1)}")
                enhanceQ.put(intm_frame)
                enhanceQ.put(frame2)
                # cv2.imshow('interpolated', intm_frame)
                # cv2.waitKey(1)
                # cv2.imshow('interpolated', frame2)

            # cv2.waitKey(1)

        print("Frame Interpolation Completed")

    except Exception as e:
        print('\nframe_iterpolation stopped due to:', e)

    del film_model

    # cv2.destroyWindow('interpolated')


def frame_enhance(enhanceQ: Queue, resultQ: Queue):
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
            with ThreadPoolExecutor() as executor:
                output_frame = executor.submit(esrgan_model.enhance, frame)
                # output_frame = future.result()
                # output_frame = esrgan_model.enhance(frame)
                print(f"Enhanced frame no. {frame_id} in {time.time() - start_time} seconds")

                # height, width, _ = frame.shape
                # output_frame = cv2.resize(output_frame, (int(width/height *480), 480))
                resultQ.put(output_frame)
                # cv2.imshow('enhanced', output_frame)
                # cv2.waitKey(1)

        print("Frame Enhancement Completed")
    except Exception as e:
        print('\nframe_enhance stopped due to:', e)
    
    del esrgan_model
    
    # cv2.destroyWindow('enhanced')


def video_output(resultQ: Queue, request_id: str):
    global fps, enhanced_video_details, total_frames
    
    # Create a directory to store the enhanced video
    output_path = f'enhance/.temp/{request_id}/video'
    os.makedirs(output_path, exist_ok=True)

    enhance_fname = f'{output_path}/enhanced.mp4'
    filename = f'{output_path}/{request_id}.mp4'

    audio_path = f'enhance/.temp/{request_id}/audio/{request_id}.m4a'

    video_out_writer = None

    try:
        frame_no = 0
        print("Writing frames to video...")
        # print(f"resultQ.empty(): {resultQ.empty()}, frame_no: {frame_no}, total_frames: {total_frames}")
        while frame_no < ((total_frames * 2) - 1) or not resultQ.empty():
            frame = resultQ.get().result()
            frame_no += 1
            # print(f"frame_no: {frame_no}")

            if video_out_writer is None:
                enhanced_video_details['shape'] = (frame.shape[1], frame.shape[0])
                video_out_writer = cv2.VideoWriter(enhance_fname, cv2.VideoWriter_fourcc(*'mp4v'), fps*2, (frame.shape[1], frame.shape[0]))

            # cv2.imwrite(f'{output_path}/frame{frame_no}.jpg', frame)
            video_out_writer.write(frame)

    except Exception as e:
        print('\nvideo_output stopped due to:', e)

    video_out_writer.release()

    # merge the audio and video
    print("Merging audio and video...")

    try:
        if os.path.exists(audio_path):
            # subprocess.run(
            #     f'ffmpeg -y -i {enhance_fname} -i {audio_path} -c:v copy -c:a copy -shortest {filename}', shell=True)

            input_video = ffmpeg.input(enhance_fname)
            input_audio = ffmpeg.input(audio_path)

            ffmpeg.concat(input_video, input_audio, v=1, a=1).output(filename).run()
            
            enhanced_video_details['url'] = upload_file.upload_file(filename)
        else:
            enhanced_video_details['url'] = upload_file.upload_file(enhance_fname)
    except Exception as e:
        print('\nAudio Merge stopped due to:', e)

    print(f"Video Enhancement Completed..!! \nEnhanced Video URL: {enhanced_video_details['url']} \nEnhanced Video Dimensions: {enhanced_video_details['shape']}")


def main(url: str, request_id: str):
    global fps, total_frames, enhanced_video_details
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)

    try:
        # cap = cv2.VideoCapture("enhance/test_videoplayback.mp4")
        cap = cv2.VideoCapture(url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                      cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f'Frame Rate: {fps} fps')
        print(f'Frame Size: {frame_size}')
        print(f"Total no. of frames: {total_frames}")

        video_duration = total_frames / fps
        print(f"Video Duration: {video_duration} seconds")
    except Exception as e:
        print(f"Exception: {e}")
        return None, "FAILED", "Video enhancement failed due to invalid url"
    
    interpolate = True
    if fps > 50:
        interpolate = False

    
    # Create a directory to store the audio
    audio_path = f'enhance/.temp/{request_id}/audio'
    os.makedirs(audio_path, exist_ok=True)

    # get the audio stream using ffmpeg
    subprocess.Popen(f'ffmpeg -y -i {url} -vn -acodec copy {audio_path}/{request_id}.m4a', shell=True)

    enhanced_video_details = {
        'url': None,
        'shape': None
    }

    frame_id = 0

    vfiQ = Queue()
    enhanceQ = Queue()
    resultQ = Queue()

    p1 = Thread(target=frame_interpolation, args=(vfiQ, enhanceQ,), daemon=True)
    p2 = Thread(target=frame_enhance, args=(enhanceQ, resultQ), daemon=True)
    p3 = Thread(target=video_output, args=(resultQ, request_id), daemon=True)

    if interpolate:
        p1.start()
    p2.start()
    p3.start()

    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if interpolate:
                vfiQ.put(frame)
            else:
                enhanceQ.put(frame)
            frame_id += 1

            # cv2.imshow('original', frame)
            # cv2.waitKey(1)
            print(frame_id, end=" ")
            # time.sleep(1/fps)

    except:
        # cv2.destroyWindow('original')
        print("Feed Ended or Error occured")

    # cv2.destroyWindow('original')

    if interpolate:
        p1.join()
    p2.join()
    p3.join()

    print(
        f"Video Duration: {video_duration} seconds, Time taken to enhance: {time.time() - start_time} seconds")
    
    print(f"Enhanced Video URL: {enhanced_video_details['url']}")

    # sys.exit(0)
    # return enhanced_video_details['url'], enhanced_video_details['shape'], "COMPLETED", "Video enhanced successfully"
    return enhanced_video_details['url'], "COMPLETED", "Video enhanced successfully"


# if __name__ == '__main__':
#     video_urls = [
#         "https://www.youtube.com/watch?v=V9DWKbalbWQ",
#         "https://www.youtube.com/watch?v=QDX-1M5Nj7s",
#         "https://www.youtube.com/watch?v=IPfo1k2JyIg",
#         "https://www.youtube.com/watch?v=a2uKphzsjMo",
#         "https://www.youtube.com/watch?v=I1J2Z_Fgado",
#     ]

#     url = video_urls[2]
#     video = pafy.new(url)
#     res = []
#     for s in video.streams:
#         # print(s.resolution, s.extension, s.get_filesize())
#         res.append(s.resolution)
#         url = s.url
#         break
#     print(sorted(res), url)

#     print(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')

#     main(url=url)
