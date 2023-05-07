import cv2
import pafy
import sys
import time
import torch
from VideoESRGAN import esrgan

video_urls = [
    "https://www.youtube.com/watch?v=V9DWKbalbWQ",
    "https://www.youtube.com/watch?v=QDX-1M5Nj7s",
    "https://www.youtube.com/watch?v=IPfo1k2JyIg",
    "https://www.youtube.com/watch?v=a2uKphzsjMo",
    "https://www.youtube.com/watch?v=I1J2Z_Fgado",
]

# cam = cv2.VideoCapture(0)

# frame
currentframe = 0

def main():
    global currentframe, video_urls

    url = video_urls[3]
    video = pafy.new(url)
    print(video)
    res = []
    for s in video.streams:
        print(s.resolution, s.extension, s.get_filesize())
        res.append(s.resolution)
        url = s.url
        # break

    print(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    print(sorted(res), url)

    cam = cv2.VideoCapture("C:/Users/Ankit Das/Desktop/videoplayback.3gp")
    fps = cam.get(cv2.CAP_PROP_FPS)
    print(f'Frame Rate: {fps}fps')


    esrgan_model = esrgan(gpu_id=0)

    while(True):
        # reading from frame
        ret,frame = cam.read()
        st = time.time()
        if ret:

            try:
                output = esrgan_model.enhance(frame)
            except RuntimeError as error:
                print('Error', error)
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    
            # writing the extracted images
            # frame = cv2.resize(frame, (1280, 686), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('test', frame)
            cv2.imshow('enhanced', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print(f'output_shape: {output.shape}, currentframe: {currentframe}, time_taken: {(time.time()-st)*1000}ms')
    
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break
  
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())