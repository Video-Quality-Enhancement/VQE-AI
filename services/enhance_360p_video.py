import time
from models import VideoEnhanceRequest
from models import EnhancedVideoResponse
from enhance import main_360p

def enhance_360p_video(video_enhance_request: VideoEnhanceRequest) -> EnhancedVideoResponse:
    print(video_enhance_request)
    
    try:
        enhanced_video_url, status, statusMessage = main_360p.main(video_enhance_request.videoUrl, video_enhance_request.requestId)
    except Exception as e:
        enhanced_video_url = None
        status = "FAILED"
        statusMessage = f"Video enhancement failed due to: {e}"

    # create response object
    enhanced_video_response = EnhancedVideoResponse(video_enhance_request, enhanced_video_url, status, statusMessage)
    return enhanced_video_response