import os, sys
from models import VideoEnhanceRequest
from producers import enhanced_video_producer
from config import AMQPconnection
import threading


def video_enhance_consumer(queue_name: str, routing_key: str, enhance_video: callable):

    connection = AMQPconnection()
    consumerCh = connection.create_channel()
    # producerCh = connection.create_channel()

    exchange = "video.enhance"
    consumerCh.exchange_declare(exchange=exchange, exchange_type="direct", durable=True)

    result = consumerCh.queue_declare(queue=queue_name, durable=True,  arguments={'x-consumer-timeout': 43200000}) # 12 hours

    consumerCh.queue_bind(exchange=exchange, queue=result.method.queue, routing_key=routing_key)

    consumerCh.basic_qos(prefetch_count=1) # 0 is no limit, 1

    def process_request(ch, delivery_tag, video_enhance_request: VideoEnhanceRequest):
        try:
            enhanced_video_response = enhance_video(video_enhance_request)
            enhanced_video_producer(enhanced_video_response)
        except Exception as e:
            print(f"Exception: {e}")
            ch.connection.add_callback_threadsafe(lambda: ch.basic_nack(delivery_tag=delivery_tag, requeue=True))
        else:
            ch.connection.add_callback_threadsafe(lambda: ch.basic_ack(delivery_tag=delivery_tag))

    def callback(ch, method, properties, body):
        delivery_tag = method.delivery_tag
        video_enhance_request = VideoEnhanceRequest.loads(body)

        process_thread = threading.Thread(target=process_request, args=(ch, delivery_tag, video_enhance_request,), daemon=True)
        process_thread.start()
                
    consumerCh.basic_consume(queue=result.method.queue, on_message_callback=callback)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    
    try:
        consumerCh.start_consuming()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    finally:
        connection.close()

# def process_request(video_enhance_request: VideoEnhanceRequest, enhance_video: callable):
#     enhanced_video_response = enhance_video(video_enhance_request)
#     enhanced_video_producer(enhanced_video_response)

# def video_enhance_consumer(queue_name: str, routing_key: str, enhance_video: callable):
#     exchange = "video.enhance"
#     connection = AMQPconnection()
#     consumerCh = connection.create_channel()
#     # producerCh = connection.create_channel()

#     stop_flag = False

#     def send_hearbeat():
#         nonlocal stop_flag
#         if stop_flag:
#             return
#         else:
#             connection.send_heartbeat()
#             threading.Timer(interval=2, function=send_hearbeat).start()

#     consumerCh.exchange_declare(exchange=exchange, exchange_type="direct", durable=True)
#     result = consumerCh.queue_declare(queue=queue_name, durable=True)

#     consumerCh.queue_bind(exchange=exchange, queue=result.method.queue, routing_key=routing_key)
#     # consumerCh.basic_qos(prefetch_count=1) # 0 is no limit, 1

#     try:
#         print(" [*] Waiting for messages. To exit press CTRL+C")
#         for method, properties, body in consumerCh.consume(queue=result.method.queue, auto_ack=False):
#             print(" [x] Received %r" % body)
#             video_enhance_request = VideoEnhanceRequest.loads(body)
#             try:
#                 # enhanced_video_response = enhance_video(video_enhance_request)
#                 # # enhanced_video_producer(enhanced_video_response)
#                 # enhance_thread = threading.Thread(target=enhanced_video_producer, args=(enhanced_video_response), daemon=True)
#                 # enhance_thread.start()
#                 # enhance_thread.join()

#                 process_thread = threading.Thread(target=process_request, args=(video_enhance_request,enhance_video,), daemon=True)
#                 process_thread.start()

#             except Exception as e:
#                 print(f"Exception: {e}")
#                 consumerCh.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
#             else:
#                 consumerCh.basic_ack(delivery_tag=method.delivery_tag)
#             finally:
#                 heartbeat_thread = threading.Thread(target=send_hearbeat, daemon=True)
#                 heartbeat_thread.start()

#                 while process_thread.is_alive():
#                     pass
#                 else:
#                     stop_flag = True
#                     heartbeat_thread.join()

#     except KeyboardInterrupt:
#         print('Interrupted')
#         connection.close()
#         try:
#             sys.exit(0)
#         except SystemExit:
#             os._exit(0)
#     finally:
#         connection.close()