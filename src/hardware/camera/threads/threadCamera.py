# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import datetime
import os
import cv2
import threading
import base64
import picamera2
import time

from src.utils.messages.allMessages import (
    mainCamera,
    serialCamera,
    Recording,
    Record,
    Brightness,
    Contrast,
    DrivingMode
)
from src.utils.messages.messageHandlerSender import messageHandlerSender
from src.utils.messages.messageHandlerSubscriber import messageHandlerSubscriber
from src.templates.threadwithstop import ThreadWithStop
from src.utils.helpers import encode_image
TAKE_SCREENSHOTS = False
class threadCamera(ThreadWithStop):
    """Thread which will handle camera functionalities.\n
    Args:
        queuesList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugger (bool): A flag for debugging.
    """

    # ================================ INIT ===============================================
    def __init__(self, queuesList, logger, debugger):
        super(threadCamera, self).__init__()
        self.queuesList = queuesList
        self.logger = logger
        self.debugger = debugger
        self.frame_rate = 30  # antes 5
        self.recording = False

        self.video_writer = ""

        self.recordingSender = messageHandlerSender(self.queuesList, Recording)
        self.mainCameraSender = messageHandlerSender(self.queuesList, mainCamera)

        self.serialCameraSender = messageHandlerSender(self.queuesList, serialCamera)
        self.frame_count = 0
        self.timestamp = datetime.datetime.now().strftime("%d-%m-%H:%M")
        self.screenshot_dir = f"screenshots/{self.timestamp}"

        if TAKE_SCREENSHOTS and not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

        self.subscribe()
        self._init_camera()
        self.Queue_Sending()
        self.Configs()

        self.act_deviation = 0.
        self.act_direction = "straight"

    def subscribe(self):
        """Subscribe function. In this function we make all the required subscribe to process gateway"""

        self.recordSubscriber = messageHandlerSubscriber(self.queuesList, Record, "lastOnly", True)
        self.brightnessSubscriber = messageHandlerSubscriber(self.queuesList, Brightness, "lastOnly", True)
        self.contrastSubscriber = messageHandlerSubscriber(self.queuesList, Contrast, "lastOnly", True)

        self.drivingMode = messageHandlerSubscriber(self.queuesList, DrivingMode, "lastOnly", True)
        

    def Queue_Sending(self):
        """Callback function for recording flag."""

        self.recordingSender.send(self.recording)
        threading.Timer(1, self.Queue_Sending).start()
        
    # =============================== STOP ================================================
    def stop(self):
        if self.recording:
            self.video_writer.release()
        super(threadCamera, self).stop()

    # =============================== CONFIG ==============================================
    def Configs(self):
        """Callback function for receiving configs on the pipe."""

        if self.brightnessSubscriber.isDataInPipe():
            message = self.brightnessSubscriber.receive()
            if self.debugger:
                self.logger.info(str(message))
            self.camera.set_controls(
                {
                    "AeEnable": False,
                    "AwbEnable": False,
                    "Brightness": max(0.0, min(1.0, float(message))),
                }
            )
        if self.contrastSubscriber.isDataInPipe():
            message = self.contrastSubscriber.receive() # de modificat marti uc camera noua 
            if self.debugger:
                self.logger.info(str(message))
            self.camera.set_controls(
                {
                    "AeEnable": False,
                    "AwbEnable": False,
                    "Contrast": max(0.0, min(32.0, float(message))),
                }
            )
        threading.Timer(1, self.Configs).start()

    # ================================ RUN ================================================
    def run(self):
        """This function will run while the running flag is True. It captures the image from camera and make the required modifies and then it send the data to process gateway."""

        send = True
        while self._running:

            #print(self.drivingMode)

            try:
                recordRecv = self.recordSubscriber.receive()
                if recordRecv is not None: 
                    self.recording = bool(recordRecv)
                    if recordRecv == False:
                        self.video_writer.release()
                    else:
                        fourcc = cv2.VideoWriter_fourcc(
                            *"XVID"
                        )  # You can choose different codecs, e.g., 'MJPG', 'XVID', 'H264', etc.
                        self.video_writer = cv2.VideoWriter(
                            "output_video" + str(time.time()) + ".avi",
                            fourcc,
                            self.frame_rate,
                            (2048, 1080),
                        )

            except Exception as e:
                print(e)

            if send:
                mainRequest = self.camera.capture_array("main")
                serialRequest = self.camera.capture_array("lores")  # Will capture an array that can be used by OpenCV library
                
                if self.recording == True:
                    self.video_writer.write(mainRequest)

                serialRequest = cv2.cvtColor(serialRequest, cv2.COLOR_YUV2BGR_I420)
                serialRequest = serialRequest[:, :320]  # Mantiene la altura y recorta el ancho
                #print(serialRequest.shape)
                if TAKE_SCREENSHOTS:
                    self.save_screenshot(serialRequest)
                
                mainEncodedImageData = encode_image(mainRequest)
                serialEncodedImageData = encode_image(serialRequest)


                self.mainCameraSender.send(mainEncodedImageData)
                self.serialCameraSender.send(serialEncodedImageData)

            send = not send

    # =============================== START ===============================================
    def start(self):
        super(threadCamera, self).start()

    # ================================ INIT CAMERA ========================================
    def _init_camera(self):
        """This function will initialize the camera object. It will make this camera object have two chanels "lore" and "main"."""

        self.camera = picamera2.Picamera2()
        config = self.camera.create_preview_configuration(
            buffer_count=1,
            queue=False,
            main={"format": "RGB888", "size": (2048, 1080)},
            lores={"size": (320, 240)},
            encode="lores",
        )
        self.camera.configure(config)
        self.camera.start()

    def save_screenshot(self,frame):
        # Guardar screenshot cada 30 frames
        if self.frame_count % 10 == 0:
            screenshot_path = os.path.join(self.screenshot_dir, f"{self.timestamp}_{self.frame_count}.png")
            cv2.imwrite(screenshot_path, frame)
        self.frame_count += 1