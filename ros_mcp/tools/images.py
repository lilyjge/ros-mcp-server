"""Image tools for ROS MCP."""

import io
import os
from pathlib import Path
from time import monotonic
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from mcp.types import ImageContent, ToolAnnotations
from PIL import Image as PILImage

# Shared image directory with Reachy Mini MCP server.
# This file lives at server/ros-mcp-server/ros_mcp/tools/images.py inside the
# reachy-mcp repo, so 5 parents up reaches the reachy-mcp root — the same base
# that Reachy's vision.py uses for its images/ directory.
_SHARED_IMAGES_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "images"
_ROS_IMAGES_DIR = _SHARED_IMAGES_DIR / "ros"


def convert_expects_image_hint(expects_image: str) -> bool | None:
    """
    Convert string-based expects_image hint to boolean for internal use.

    Args:
        expects_image (str): String hint about whether to expect image data
            - "true": prioritize image parsing
            - "false": skip image detection for faster processing
            - "auto": auto-detect based on message fields (default)
            - any other value: treated as "auto"

    Returns:
        bool | None: Converted hint for parse_input function
            - True: prioritize image parsing
            - False: skip image detection
            - None: auto-detect
    """
    if expects_image == "true":
        return True
    elif expects_image == "false":
        return False
    else:  # "auto" or any other value
        return None


def _encode_image_to_imagecontent(image) -> ImageContent:
    """
    Encodes a PIL Image to a format compatible with ImageContent.

    Args:
        image (PIL.Image.Image): The image to encode.

    Returns:
        ImageContent: JPEG-encoded image wrapped in an ImageContent object.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    img_obj = Image(data=img_bytes, format="jpeg")
    return img_obj.to_image_content()


def register_image_tools(
    mcp: FastMCP,
    ws_manager=None,
    default_camera_topic: str = "/image_raw/compressed",
) -> None:
    """Register all image-related tools.
    
    Args:
        mcp: FastMCP instance to register tools with
        ws_manager: WebSocketManager instance (optional, needed for capture_camera_image)
        default_camera_topic: Default camera topic for capture_camera_image
    """

    @mcp.tool(
        description=(
            "Capture an image from a ROS camera topic.\n"
            "This is a convenience tool that subscribes to a camera topic, "
            "captures one frame, saves it to a shared directory, and returns the path.\n\n"
            "By default (return_image_content=False), the image is saved to a shared directory "
            "and the relative_path is returned. Pass this relative_path to Reachy's describe_image "
            "tool to get a text description of what the robot sees.\n\n"
            "Example workflow:\n"
            "  1. result = capture_camera_image()  # saves image, returns relative_path\n"
            "  2. description = describe_image(result['relative_path'])  # Reachy describes it\n\n"
            "capture_camera_image()  # Uses default topic, returns path\n"
            "capture_camera_image(topic='/camera/rgb/image_raw', msg_type='sensor_msgs/msg/Image')\n"
            "capture_camera_image(return_image_content=True)  # Returns raw image bytes instead\n"
        ),
        annotations=ToolAnnotations(
            title="Capture Camera Image",
            readOnlyHint=True,
        ),
    )
    def capture_camera_image(
        topic: str = default_camera_topic,
        msg_type: str = "auto",
        return_image_content: bool = False,
        timeout: float = 5.0,
    ) -> dict | ImageContent:  # type: ignore  # See issue #140
        """
        Capture an image from a ROS camera topic.
        
        This tool subscribes to a camera topic, captures one frame, saves it to a shared 
        directory, and optionally returns the image for immediate analysis by the LLM.
        
        Images are saved with timestamped filenames in a shared directory that can be 
        accessed by other MCP servers (e.g., Reachy Mini's describe_image tool).
        
        Args:
            topic (str): Camera topic name (default: "/image_raw/compressed")
            msg_type (str): Message type. Use "auto" to infer raw vs compressed
                from the topic name. (default: "auto")
            return_image_content (bool): If True, return the image for LLM analysis. 
                If False, save the image and return the relative_path for use with 
                Reachy's describe_image tool. (default: False)
            timeout (float): Timeout in seconds to wait for an image (default: 5.0)
        
        Returns:
            ImageContent or dict: The captured image for multimodal LLMs, or a dict with:
                - message: Success message
                - image_path: Absolute path to saved image
                - relative_path: Relative path from Reachy's images directory (use this with describe_image)
        """
        if ws_manager is None:
            return {"error": "WebSocketManager not available. This tool requires a ROS connection."}

        if msg_type in ("", "auto", None):
            msg_type = (
                "sensor_msgs/msg/CompressedImage"
                if "compressed" in topic.lower()
                else "sensor_msgs/msg/Image"
            )
        
        # Import here to avoid circular dependency
        import time
        
        # Construct the rosbridge subscribe message
        subscribe_msg: dict = {
            "op": "subscribe",
            "topic": topic,
            "type": msg_type,
            "queue_length": 1,
            "throttle_rate": 0,
        }
        
        # Subscribe and wait for the first message
        with ws_manager:
            # Send subscription request
            send_error = ws_manager.send(subscribe_msg)
            if send_error:
                return {"error": f"Failed to subscribe to {topic}: {send_error}"}
            
            # Loop until we receive the first message or timeout
            end_time = time.time() + timeout
            while time.time() < end_time:
                response = ws_manager.receive(timeout=0.5)
                if response is None:
                    continue
                
                # Parse the response with image hint
                from ros_mcp.utils.websocket import parse_input
                msg_data, was_parsed_as_image = parse_input(response, expects_image=True)
                
                if not msg_data:
                    continue
                
                # Check for status errors
                if msg_data.get("op") == "status" and msg_data.get("level") == "error":
                    unsubscribe_msg = {"op": "unsubscribe", "topic": topic}
                    ws_manager.send(unsubscribe_msg)
                    return {"error": f"Rosbridge error: {msg_data.get('msg', 'Unknown error')}"}
                
                # Check for the image message
                if msg_data.get("op") == "publish" and msg_data.get("topic") == topic:
                    # Unsubscribe
                    unsubscribe_msg = {"op": "unsubscribe", "topic": topic}
                    ws_manager.send(unsubscribe_msg)
                    
                    # If image was parsed and saved, move it to the shared directory
                    if was_parsed_as_image:
                        temp_path = "./camera/received_image.jpeg"
                        
                        # Create timestamped filename in shared Reachy images directory
                        _ROS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                        timestamp = monotonic()
                        shared_image_path = _ROS_IMAGES_DIR / f"ros_{timestamp:.3f}.jpg"
                        
                        # Move image to shared directory
                        if os.path.exists(temp_path):
                            import shutil
                            shutil.move(temp_path, str(shared_image_path))
                            
                            if return_image_content:
                                img = PILImage.open(shared_image_path)
                                return _encode_image_to_imagecontent(img)
                            else:
                                return {
                                    "message": f"Image captured from {topic} and saved to {shared_image_path}",
                                    "image_path": str(shared_image_path),
                                    "relative_path": f"ros/ros_{timestamp:.3f}.jpg",
                                }
                        else:
                            return {"error": "Image was received but failed to save"}
                    else:
                        return {"error": f"Received message from {topic} but it was not recognized as an image"}
            
            # Timeout - unsubscribe and return error
            unsubscribe_msg = {"op": "unsubscribe", "topic": topic}
            ws_manager.send(unsubscribe_msg)
            return {"error": f"Timeout waiting for image from {topic} after {timeout} seconds"}

    @mcp.tool(
        description=(
            "Analyze a previously received image that was saved by any ROS operation.\n"
            "Images can be received from:\n"
            "- Any topic containing image data (not just topics with 'Image' in the name)\n"
            "- Service responses containing image data\n"
            "- subscribe_once() or subscribe_for_duration() operations\n"
            "- capture_camera_image() tool\n"
            "Use this tool to analyze the saved image after receiving it from any source.\n"
        ),
        annotations=ToolAnnotations(
            title="Analyze Previously Received Image",
            readOnlyHint=True,
        ),
    )
    def analyze_previously_received_image(
        image_path: str = "./camera/received_image.jpeg",
    ) -> ImageContent:  # type: ignore  # See issue #140
        """
        Analyze the previously received image saved at the specified path.

        This tool loads the previously saved image from the specified path
        (which can be created by any ROS operation that receives image data), and converts
        it into an MCP-compatible ImageContent format so that the LLM can interpret it.

        Images can be received from:
        - Any Topic containing image data
        - Any Service responses containing image data
        - subscribe_once() or subscribe_for_duration() operations
        - capture_camera_image() tool

        Args:
            image_path (str): Path to the saved image file (default: "./camera/received_image.jpeg")

        Returns:
            ImageContent: JPEG-encoded image wrapped in an ImageContent object, or error dict if file not found.
        """
        if not os.path.exists(image_path):
            return {"error": f"No image found at {image_path}"}  # type: ignore[return-value]  # See issue #140
        img = PILImage.open(image_path)
        return _encode_image_to_imagecontent(img)
