"""
Stitch two opposing 180-degree fisheye images into a single equirectangular
panorama for visual SLAM.

Subscribes:
  /world/<world>/model/rover_360cam/.../fisheye_front/image  (sensor_msgs/Image, R8G8B8)
  /world/<world>/model/rover_360cam/.../fisheye_rear/image

Publishes:
  /rover_360cam/equirect  (sensor_msgs/Image, R8G8B8, W=1600 H=800)

Geometry assumption: rear cam is rotated 180 deg about Z relative to front,
so its image covers azimuth [-180, 0) and front covers [0, 180). We project
each fisheye to its half of the equirectangular sphere with a precomputed
LUT and merge with a small overlap blend.

Equirect convention:
  u in [0, W) -> azimuth phi  in [-pi, +pi)
  v in [0, H) -> elevation th in [+pi/2, -pi/2]

For each equirect pixel we compute its 3D ray, rotate into the camera frame,
and use the equidistant fisheye model: r = f * theta_from_optical_axis.
"""
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


def build_lut(eq_w: int, eq_h: int, fisheye_size: int,
              azimuth_center: float, fov: float):
    """LUT mapping equirect (u,v) -> fisheye (x,y) for one camera.

    azimuth_center: where this camera's optical axis points, in equirect azimuth.
    fov: full FOV of fisheye in radians (e.g. pi for 180-deg).
    Returns (mapx, mapy, mask) all of shape (eq_h, eq_w).
    """
    u = np.arange(eq_w)
    v = np.arange(eq_h)
    uu, vv = np.meshgrid(u, v)
    phi = (uu / eq_w) * 2.0 * np.pi - np.pi          # [-pi, pi)
    theta = (0.5 - vv / eq_h) * np.pi                # [+pi/2, -pi/2]

    # 3D ray in world (camera-rig) frame: x fwd, y left, z up
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    # Rotate so this camera's optical axis aligns with +X
    cphi = np.cos(-azimuth_center)
    sphi = np.sin(-azimuth_center)
    xr = cphi * x - sphi * y
    yr = sphi * x + cphi * y
    zr = z

    # Angle from optical axis (+X)
    ang = np.arctan2(np.sqrt(yr * yr + zr * zr), xr)
    mask = ang <= (fov * 0.5)

    # Equidistant model: r = f * ang, normalized so r=1 at ang=fov/2
    r = ang / (fov * 0.5)
    az_in_image = np.arctan2(zr, yr)  # roll about optical axis
    # Image x-right, y-down convention; map +z (up) to -y_img
    fx = 0.5 * fisheye_size + 0.5 * fisheye_size * r * np.cos(az_in_image)
    fy = 0.5 * fisheye_size - 0.5 * fisheye_size * r * np.sin(az_in_image)

    return fx.astype(np.float32), fy.astype(np.float32), mask


class FisheyeToEquirect(Node):
    def __init__(self):
        super().__init__('fisheye_to_equirect')
        self.declare_parameter('front_topic', '/fisheye_front/image')
        self.declare_parameter('rear_topic', '/fisheye_rear/image')
        self.declare_parameter('out_topic', '/rover_360cam/equirect')
        self.declare_parameter('eq_width', 1600)
        self.declare_parameter('eq_height', 800)
        self.declare_parameter('fisheye_size', 800)
        self.declare_parameter('fov_deg', 180.0)

        eq_w = self.get_parameter('eq_width').value
        eq_h = self.get_parameter('eq_height').value
        fsize = self.get_parameter('fisheye_size').value
        fov = np.deg2rad(self.get_parameter('fov_deg').value)

        # Front cam optical axis at azimuth = 0; rear at azimuth = pi.
        self.fx_f, self.fy_f, self.mask_f = build_lut(eq_w, eq_h, fsize, 0.0, fov)
        self.fx_r, self.fy_r, self.mask_r = build_lut(eq_w, eq_h, fsize, np.pi, fov)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(
            Image, self.get_parameter('out_topic').value, 10)

        sub_f = message_filters.Subscriber(
            self, Image, self.get_parameter('front_topic').value)
        sub_r = message_filters.Subscriber(
            self, Image, self.get_parameter('rear_topic').value)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [sub_f, sub_r], queue_size=5, slop=0.05, allow_headerless=True)
        self.sync.registerCallback(self.cb)
        self.get_logger().info(
            f'fisheye_to_equirect ready ({eq_w}x{eq_h}, fisheye={fsize}, fov={np.rad2deg(fov):.0f} deg)')

    def cb(self, msg_f: Image, msg_r: Image):
        import cv2
        img_f = self.bridge.imgmsg_to_cv2(msg_f, desired_encoding='rgb8')
        img_r = self.bridge.imgmsg_to_cv2(msg_r, desired_encoding='rgb8')

        warp_f = cv2.remap(img_f, self.fx_f, self.fy_f,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warp_r = cv2.remap(img_r, self.fx_r, self.fy_r,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        out = np.zeros_like(warp_f)
        out[self.mask_f] = warp_f[self.mask_f]
        out[self.mask_r] = warp_r[self.mask_r]
        # Simple overlap average where both masks are valid
        both = self.mask_f & self.mask_r
        if both.any():
            out[both] = (warp_f[both].astype(np.uint16)
                        + warp_r[both].astype(np.uint16)) // 2

        out_msg = self.bridge.cv2_to_imgmsg(out, encoding='rgb8')
        out_msg.header.stamp = msg_f.header.stamp
        out_msg.header.frame_id = 'rover_360cam_optical'
        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = FisheyeToEquirect()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
