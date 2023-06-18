from dotenv import load_dotenv
from handlers import video_enhance_handler_720p

def main():
    # * can add logger and stuff here
    load_dotenv() 
    video_enhance_handler_720p()

if __name__ == '__main__':
    main()