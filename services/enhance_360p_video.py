import time
from models import VideoEnhanceRequest
from models import EnhancedVideoResponse
from enhance import main_360p

# def main_360p(video_enhance_request: VideoEnhanceRequest):
#     print("main 360p")
#     time.sleep(2)
#     return "https://download.samplelib.com/mp4/sample-5s.mp4", "success", "Video enhanced successfully"

def enhance_360p_video(video_enhance_request: VideoEnhanceRequest) -> EnhancedVideoResponse:
    print(video_enhance_request)
    
    # perform action here
    # time.sleep(2)
    try:
        enhanced_video_url, status, statusMessage = main_360p.main(video_enhance_request.videoUrl, video_enhance_request.requestId)
    except Exception as e:
        enhanced_video_url = None
        status = "failed"
        statusMessage = f"Video enhancement failed due to: {e}"

    # set output variables
    # enhanced_video_url = "https://download.samplelib.com/mp4/sample-5s.mp4"
    # status = "success"
    # statusMessage = "Video enhanced successfully"

    # create response object
    enhanced_video_response = EnhancedVideoResponse(video_enhance_request, enhanced_video_url, status, statusMessage)
    return enhanced_video_response