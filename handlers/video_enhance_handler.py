from consumers import video_enhance_consumer
from services import enhance_144p_video, enhance_240p_video, enhance_360p_video, enhance_480p_video, enhance_720p_video

def video_enhance_handler_720p():
    queue = "720p_queue"
    routing_key = "720p"
    video_enhance_consumer(queue, routing_key, enhance_720p_video)

def video_enhance_handler_480p():
    queue = "480p_queue"
    routing_key = "480p"
    video_enhance_consumer(queue, routing_key, enhance_480p_video)

def video_enhance_handler_360p():
    queue = "360p_queue"
    routing_key = "360p"
    video_enhance_consumer(queue, routing_key, enhance_360p_video)

def video_enhance_handler_240p():
    queue = "360p_queue"
    routing_key = "360p"
    video_enhance_consumer(queue, routing_key, enhance_240p_video)

def video_enhance_handler_144p():
    queue = "360p_queue"
    routing_key = "360p"
    video_enhance_consumer(queue, routing_key, enhance_144p_video)
