#!/usr/bin/env python3
"""OAK-D measurement visualization & acquisition script.
Provides visualization for the most important building blocks of the
OAK-D StereoDepth node, as well as reconstructed point cloud.
Disparity map and rectified images can visualized, and point cloud can be 
indeptendantly projected from both.
Point clouds can be captured. 
Disparity and rectification are all computed on the device!

Can use one or more additional depth modes (LRCHECK, EXTENDED, SUBPIXEL).
Depth modes are extra computations performed during the depth computation.
If some of these additional depth modes are enabled:
 - depth output is FP16, not U16.
 - median filtering is disabled on device.
 - with SUBPIXEL, either depth or disparity has valid data.

Otherwise, depth output is U16 (mm) and median is functional.
But like on Gen1 OAK-D, either depth or disparity has valid data. Work on this is in
Luxonis's pipeline.
"""
from generate_pc import PointCloudGenerator
import cv2
import numpy as np
import depthai
from time import sleep
import datetime
import argparse
from typing import Type
from typing import Tuple
from typing import List
from typing import Optional

# Camera intrinsics (right camera is used):
RIGHT_INTRINSICS = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]

# TODO(vice) Move to YAML. Following config is the best for close-range measurement.
# Configure depthai StereoDepth node:
OUT_DEPTH = False  # Output depth. Disparity by default.
OUT_RECTIFIED = True  # Output and display rectified streams.
LRCHECK = True  # Better handling of occlusions.
EXTENDED = True  # Closer-in minimum depth, disparity range is doubled.
SUBPIXEL = False  # Better accuracy for longer distance, fractional disparity 32-levels
CONFIDENCE_THRESHOLD = 200


def create_stereo_depth_pipeline(
    out_depth: bool,
    out_rectified: bool,
    median: Type[depthai.StereoDepthProperties.MedianFilter],
    lrcheck: bool,
    extended: bool,
    subpixel: bool,
    confidence_threshold: int,
    from_camera: bool = True,
) -> Tuple[Type[depthai.Pipeline], list]:
    """Creates a stereo depth pipeline object with the corresponding data streams.
    Pipeline consists of the pipeline object with nodes and connections
    defined inside of it.
    Most of the processing is done on the camera itself. To get the data in or
    out from the camera, XLink node has to be used which provides input and
    output streams.

    Args:
        out_depth: If true, depth will be outputed. Disparity by default.
        out_rectified: If true, rectified streams will be outputed.
        median: Median filter to be used.
        lrcheck: If true, left - right check will be done.
        extended: If true, extended disparity will be used.
        subpixel: If true, subpixel computation of disparity will be used.
        confidence_threshold: Confidence from 0 (highest) to 255 (lowest) that determines
                              will the disparity computed for a given pixel be taken
                              into account.
        from_camera: If true, device will be used as a source. Otherwise, images from
        a directory will be used.

    Returns:
        Pipeline object and list of stream names.
    """
    print("Creating Stereo Depth pipeline: ", end="")
    if from_camera:
        print("MONO CAMS -> STEREO -> XLINK OUT")
    else:
        print("XLINK IN -> STEREO -> XLINK OUT")
    pipeline = depthai.Pipeline()

    # Define the basic nodes.
    if from_camera:
        cam_left = pipeline.createMonoCamera()
        cam_right = pipeline.createMonoCamera()
    else:
        cam_left = pipeline.createXLinkIn()
        cam_right = pipeline.createXLinkIn()

    stereo = pipeline.createStereoDepth()

    if from_camera:
        cam_left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
        cam_right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)
        for cam in [cam_left, cam_right]:  # Common config.
            cam.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_720_P)
    else:
        cam_left.setStreamName("in_left")
        cam_right.setStreamName("in_right")

    # Configure the stereo node.
    stereo.setOutputDepth(out_depth)
    stereo.setOutputRectified(out_rectified)
    stereo.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout.
    stereo.setMedianFilter(median)  # KERNEL_7x7 default.
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)

    # Increase the accuracy or filling rate of the depth measurement.
    # See: https://docs.luxonis.com/projects/api/en/latest/tutorials/configuring-stereo-depth/
    # ?highlight=stereo#stereo-depth-confidence-threshold
    stereo.setConfidenceThreshold(confidence_threshold)

    if from_camera:
        # Default: EEPROM calib is used, and resolution taken from MonoCamera nodes.
        # stereo.loadCalibrationFile(path) can also be used.
        pass
    else:
        stereo.setEmptyCalibration()  # Set if the input frames are already rectified.
        stereo.setInputResolution(1280, 720)

    # Define input & output nodes to the host.
    xout_left = pipeline.createXLinkOut()
    xout_right = pipeline.createXLinkOut()
    xout_depth = pipeline.createXLinkOut()
    xout_disparity = pipeline.createXLinkOut()
    xout_rectif_left = pipeline.createXLinkOut()
    xout_rectif_right = pipeline.createXLinkOut()

    # Define input & output streams.
    xout_left.setStreamName("left")
    xout_right.setStreamName("right")
    xout_depth.setStreamName("depth")
    xout_disparity.setStreamName("disparity")
    xout_rectif_left.setStreamName("rectified_left")
    xout_rectif_right.setStreamName("rectified_right")

    # Link the nodes (& streams).
    cam_left.out.link(stereo.left)
    cam_right.out.link(stereo.right)
    stereo.syncedLeft.link(xout_left.input)
    stereo.syncedRight.link(xout_right.input)
    stereo.depth.link(xout_depth.input)
    stereo.disparity.link(xout_disparity.input)
    stereo.rectifiedLeft.link(xout_rectif_left.input)
    stereo.rectifiedRight.link(xout_rectif_right.input)

    streams = ["left", "right"]
    if OUT_RECTIFIED:
        streams.extend(["rectified_left", "rectified_right"])
    streams.extend(["disparity", "depth"])

    return pipeline, streams


def convert_to_cv2_frame(
    name: str,
    image: Type[depthai.ImgFrame],
    RIGHT_INTRINSICS: List[List],
    extended: bool,
    subpixel: bool,
    color_disparity: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Converts the output stream image frames from the device to OpenCV frames, ready for visualization.
    Functions will perform different operations depending on the type of the frame that it
    receives from the device. If disparity frame is received, depth will also be computed and added
    to the output.
    N.B. CPU heavy operation!

    Args:
        name: Name of the output stream.
        image: Image acquired from the output queue from the device.
        RIGHT_INTRINSICS: Camera intrinsics.
        extended: If true, disparity computation will be extended.
        subpixel: If true, subpixel disparity computation will be used.
        color_disparity: If true, a color map will be applied to the frame containing disparity data.

    Returns:
        Frame array, ready for visualization, and depth, if available.
    """
    # Set basic transformation parameters.
    baseline = 75  # mm
    focal = RIGHT_INTRINSICS[0][0]
    max_disp = 96
    disp_type = np.uint8
    disp_levels = 1

    depth = None

    if extended:
        max_disp *= 2
    if subpixel:
        max_disp *= 32
        disp_type = np.uint16  # 5 bits fractional disparity.
        disp_levels = 32

    # Unpack the image data, depending on the type.
    # N.B. Possible improvement: check image frame type instead of name.
    data, w, h = image.getData(), image.getWidth(), image.getHeight()
    if name == "rgb_preview":
        frame = np.array(data).reshape((3, h, w)).transpose(1, 2, 0).astype(np.uint8)
    elif name == "rgb_video":  # YUV NV12
        yuv = np.array(data).reshape((h * 3 // 2, w)).astype(np.uint8)
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    elif name == "depth":
        # This contains FP16 with (LRCHECK or EXTENDED or SUBPIXEL).
        frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
    elif name == "disparity":
        disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))
        # Compute depth from disparity (32 levels).
        with np.errstate(divide="ignore"):  # Should be safe to ignore div by zero here.
            depth = (disp_levels * baseline * focal / disp).astype(np.uint16)

        # Extend disparity range to better visualize it (optional).
        frame = (disp * 255.0 / max_disp).astype(np.uint8)
        # Apply a color map (optional).
        if color_disparity:
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)

    else:  # Mono streams / single channel.
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        if name.startswith("rectified_"):
            frame = cv2.flip(frame, 1)
        if name == "rectified_right":
            global last_rectif_right
            last_rectif_right = frame

    return frame, depth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pc_disparity",
        help="Convert and visualize point cloud from the disparity map.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-pc_rectified",
        help="Convert and visualize point cloud from the rectified (righ) image.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-static",
        default=False,
        action="store_true",
        help="Run stereo on static frames passed from host 'dataset' folder.",
    )
    parser.add_argument(
        "-show_disparity",
        default=False,
        action="store_true",
        help="Enables visualization of the disparity map.",
    )
    parser.add_argument(
        "-show_rectified",
        default=False,
        action="store_true",
        help="Enables visualization of the L and R rectified images.",
    )
    args = parser.parse_args()

    # Determine the fram source: device or folder.
    source_camera = not args.static

    # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7
    median = depthai.StereoDepthProperties.MedianFilter.KERNEL_7x7

    # Sanitize some incompatible options as median filter can't work with them.
    if LRCHECK or EXTENDED or SUBPIXEL:
        median = depthai.StereoDepthProperties.MedianFilter.MEDIAN_OFF

    # Print out the basic setup:
    print("Starting OAK-D measurement visualization & acquisition.")
    print("General parameters:")
    print("    Camera source:  ", source_camera)
    print("    Pc from disparity:", args.pc_disparity)
    print("    Pc from rectified:", args.pc_rectified)
    print("StereoDepth node configuration options:")
    print("    Left-Right check:  ", LRCHECK)
    print("    Extended disparity:", EXTENDED)
    print("    Subpixel:          ", SUBPIXEL)
    print("    Median filtering:  ", median)

    pc_converter = None
    if args.pc_disparity or args.pc_rectified:
        if OUT_RECTIFIED:
            pc_converter = PointCloudGenerator(RIGHT_INTRINSICS, 1280, 720)
            print(
                "Point Cloud Generation started! Press q to exit. Press c to capture"
                "measurement."
            )
        else:
            print(
                "Point Cloud Generation will not be provided, as OUT_RECTIFIED is not set"
            )

    # Define a pipeline.
    pipeline, streams = create_stereo_depth_pipeline(
        OUT_DEPTH,
        OUT_RECTIFIED,
        median,
        LRCHECK,
        EXTENDED,
        SUBPIXEL,
        CONFIDENCE_THRESHOLD,
        source_camera,
    )

    with depthai.Device(pipeline) as device:
        print("Starting pipeline")
        device.startPipeline()
        # Define stream queues - required to send & received data to & from a device.
        # Define static frame .png dataset queues.
        in_streams = []
        if not source_camera:
            # Reversed order trick:
            # The sync stage on device side has a timeout between receiving left
            # and right frames. In case a delay would occur on host between sending
            # left and right, the timeout will get triggered.
            # One needs to make sure to send first the right frame, then left.
            in_streams.extend(["in_right", "in_left"])
        in_q_list = []
        for s in in_streams:
            q = device.getInputQueue(s)
            in_q_list.append(q)

        # Define output device queues.
        q_list = []
        for s in streams:
            q = device.getOutputQueue(s, 8, blocking=False)
            q_list.append(q)

        # A timestamp is required for input frames, for the sync stage in Stereo node.
        timestamp_ms = 0
        index = 0
        while True:
            # Handle input streams, if any (static dataset).
            if in_q_list:
                dataset_size = 2  # Number of image pairs.
                frame_interval_ms = 33
                for q in in_q_list:
                    name = q.getName()
                    path = "dataset/" + str(index) + "/" + name + ".png"
                    data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(720 * 1280)
                    tstamp = datetime.timedelta(
                        seconds=timestamp_ms // 1000, milliseconds=timestamp_ms % 1000
                    )
                    img = depthai.ImgFrame()
                    img.setData(data)
                    img.setTimestamp(tstamp)
                    img.setWidth(1280)
                    img.setHeight(720)
                    q.send(img)
                    print(
                        "Sent frame: {:25s}".format(path), "timestamp_ms:", timestamp_ms
                    )
                timestamp_ms += frame_interval_ms
                index = (index + 1) % dataset_size
                # Insert delay between iterations, host driven pipeline (optional).
                sleep(frame_interval_ms / 1000)

            # Handle & visualize output streams from the device.
            for q in q_list:
                name = q.getName()
                image = q.get()
                # Skip some streams, to reduce CPU load
                if name in ["left", "right", "depth"]:
                    continue
                frame, _ = convert_to_cv2_frame(
                    name, image, RIGHT_INTRINSICS, EXTENDED, SUBPIXEL, False
                )
                if args.show_disparity and name == "disparity":
                    cv2.imshow(name, frame)
                if args.show_disparity and (
                    name == "rectified_left" or name == "rectified_right"
                ):
                    cv2.imshow(name, frame)

                # Visualize projected point cloud.
                if pc_converter is not None and name == "disparity":
                    frame, depth = convert_to_cv2_frame(
                        name, image, RIGHT_INTRINSICS, EXTENDED, SUBPIXEL, False
                    )
                    # Project disparity to the pc.
                    if args.pc_disparity:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pc_converter.rgbd_to_pc(depth, frame_rgb)
                    # Project rectified right to the pc.
                    if args.pc_rectified:
                        pc_converter.rgbd_to_pc(depth, last_rectif_right)
                    pc_converter.visualize_pc()

            if cv2.waitKey(1) == ord("q"):
                if pc_converter:
                    pc_converter.close_window()
                break
