import struct
import threading
import socket
import os

# CRC校验表
crc32_tab = [
    0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
    0xe963a535, 0x9e6495a3, 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
    0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
    0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
    0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
    0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x01db7106,
    0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
    0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
    0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
    0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
    0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
    0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
    0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
    0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
    0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
    0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
    0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
    0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
    0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
    0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
    0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
    0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
    0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d]


# crc校验函数
def crc32(data: bytes, crc: int = 1) -> int:
    for byte in data:
        crc = crc32_tab[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc


# IMU状态类
class ImuData:
    def __init__(self):
        self.header1 = 0
        self.header2 = 0
        self.sequence = 0
        self.data_type = 0
        self.accelerometer = [0.0, 0.0,
                              0.0]  # x y z方向上的加速度单位为m/s2   ax: accelerometer[0]  ay: accelerometer[1]  az: accelerometer[2]
        self.gyroscope = [0.0, 0.0, 0.0]  # x y Z方向上的角速度单位为rad/s  wx: gyroscope[0]  wy: gyroscope[1]  wz: gyroscope[2]
        self.quaternion = [1.0, 0.0, 0.0,
                           0.0]  # w x y z归一化后的四元数         qw: quaternion[0] qx: quaternion[1] qy: quaternion[2] qz: quaternion[3]
        self.rpy = [0.0, 0.0, 0.0]  # 欧拉角                       roll: rpy[0]  pitch: rpy[1]  yaw: rpy[2]
        self.temperature = 0  # 温度
        self.timestamp = 0  # 时间戳
        self.crc_32 = 0


# 里程计状态类
class OdometerData:
    def __init__(self):
        self.header1 = 0
        self.header2 = 0
        self.sequence = 0
        self.data_type = 0
        self.position_x = 0.0
        self.position_y = 0.0
        self.position_z = 0.0
        self.orientation_x = 0.0
        self.orientation_y = 0.0
        self.orientation_z = 0.0
        self.orientation_w = 0.0
        self.linear_x = 0.0
        self.linear_y = 0.0
        self.linear_z = 0.0
        self.angular_x = 0.0
        self.angular_y = 0.0
        self.angular_z = 0.0
        self.crc_32 = 0


# 电机状态类
class MotorState:
    def __init__(self, mode=0, pos=0.0, w=0.0, t=0.0, temperature=0):
        self.mode = mode  # 电机模式 0失能 10使能
        self.pos = pos  # 电机角度 单位为rad
        self.w = w  # 电机角速度 单位为rad/s
        self.t = t  # 电机转矩 单位为NM
        self.temperature = temperature  # 电机温度 单位为摄氏度


# 机器人电机状态帧类
class MotorStateData:
    def __init__(self, motor_count=12):
        self.header1 = 0  # 帧头1
        self.header2 = 0  # 帧头2
        self.sequence = 0  # 帧序号
        self.data_type = 0  # 帧类型
        self.state = [MotorState() for _ in range(motor_count)]  # motor_count个电机状态
        self.crc_32 = 0  # CRC校验码


# 四足电机顺序
# 0: FR-a   1: FR-h   2: FR-k
# 3: FL-a   4: FL-h   5: FL-k
# 6: BR-a   7: BR-h   8: BR-k
# 9: BL-a   10: BL-h  11: BL-k
# "torso_link", "neck_link",
# "left_hip_yaw_link", "left_hip_roll_link", "left_hip_pitch_link", "left_knee_link", "left_ankle_link", "left_toe_link", // left leg
# "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link", "right_knee_link", "right_ankle_link", "right_toe_link", // right leg
# "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_pitch_link",
# "left_elbow_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link", "left_gripper_left_link", "left_gripper_right_link" // left arm
# "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_pitch_link",
# "right_elbow_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link", "right_gripper_left_link", "right_gripper_right_link" // right arm
class JointPosition:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x  # 关节位置X坐标
        self.y = y  # 关节位置Y坐标
        self.z = z  # 关节位置Z坐标


class JointOrientation:
    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0):
        self.w = w  # 方向W
        self.x = x  # 方向X
        self.y = y  # 方向Y
        self.z = z  # 方向Z


class JointState:
    def __init__(self):
        self.position = JointPosition()  # 关节位置
        self.orientation = JointOrientation()  # 关节方向


class JointStateData:
    def __init__(self, motor_count=32):
        self.header1 = 0  # 帧头1
        self.header2 = 0  # 帧头2
        self.sequence = 0  # 帧序号
        self.data_type = 0  # 帧类型
        self.state = [JointState() for _ in range(motor_count)]  # motor_count个关节状态
        self.crc_32 = 0


class RemoteData:
    def __init__(self):
        self.header1 = 0  # 帧头1 (uint8_t)
        self.header2 = 0  # 帧头2 (uint8_t)
        self.sequence = 0  # 序列号 (uint16_t)
        self.data_type = 0  # 数据类型 (uint32_t)
        self.remote_type = 0  # 遥控器类型 (uint16_t) 适配 AT9S 遥控器
        self.swa = 0  # 按键A (uint8_t), 按键上拨:0 按键下拨:1
        self.swb = 0  # 按键B (uint8_t), 按键上拨:0 按键下拨:1
        self.swe = 0  # 按键E (uint8_t), 按键上拨:0 按键中拨:1 按键下拨:2
        self.swg = 0  # 按键G (uint8_t), 按键上拨:0 按键中拨:1 按键下拨:2
        self.reserved1 = 0  # 按键预留1 (uint8_t)
        self.reserved2 = 0  # 按键预留2 (uint8_t)
        self.rocker_left_x = 0.0  # 左摇杆水平 (float) 范围: -1~1 摇杆中间值为0
        self.rocker_left_y = 0.0  # 左摇杆垂直 (float) 范围: -1~1 摇杆中间值为0
        self.rocker_right_x = 0.0  # 右摇杆水平 (float) 范围: -1~1 摇杆中间值为0
        self.rocker_right_y = 0.0  # 右摇杆垂直 (float) 范围: -1~1 摇杆中间值为0
        self.crc_32 = 0  # 校验码 (uint32_t)


# 机器人数据接收类
class DataReceiver:
    # 类初始化函数
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建socket
        server_address = ('192.168.93.107', 8787)  # 0.0.0.0 表示绑定到所有可用接口
        self.sock.bind(server_address)

        self.imu_data = ImuData()
        self.motor_state_data = MotorStateData(12)
        self.left_arm_state_data = MotorStateData(8)
        self.right_arm_state_data = MotorStateData(8)
        self.odometer_data = OdometerData()
        self.joint_state_data = JointStateData(32)
        self.remote_data = RemoteData()

        # 创建一个锁对象
        self.imu_data_lock = threading.Lock()
        self.motor_data_lock = threading.Lock()
        self.arm_data_lock = threading.Lock()
        self.odometer_data_lock = threading.Lock()
        self.joint_state_data_lock = threading.Lock()
        self.remote_data_lock = threading.Lock()
        self.recv_thread = threading.Thread(target=self.receive_and_process_data)  # 创建数据接收线程
        self.recv_thread.start()  # 启动线程

    # IMU数据解析函数
    def parse_imu_data(self, data: bytes) -> ImuData:
        imu_format = 'BBHI3f3f4f3fI2I'  # 定义imu数据帧格式
        crc_format = 'I'

        data_body = data[:-4]  # 除去最后的 CRC32 校验码
        received_crc = struct.unpack(crc_format, data[-4:])[0]  # 使用小端序解包 CRC32

        computed_crc = crc32(data_body)
        # 调试输出
        # print(f"数据体: {data.hex()}")
        # print(f"接收到的 CRC32: {received_crc:08X}")
        # print(f"计算的 CRC32: {computed_crc:08X}")
        if computed_crc != received_crc:
            raise ValueError(f"CRC32 校验失败: 计算值 {computed_crc:08X}, 接收到的值 {received_crc:08X}")

        unpacked_data = struct.unpack(imu_format, data)  # 按格式解析传入的数据

        imu_data = ImuData()
        imu_data.header1 = unpacked_data[0]
        imu_data.header2 = unpacked_data[1]
        imu_data.sequence = unpacked_data[2]
        imu_data.data_type = unpacked_data[3]
        imu_data.accelerometer = unpacked_data[4:7]
        imu_data.gyroscope = unpacked_data[7:10]
        imu_data.quaternion = unpacked_data[10:14]
        imu_data.rpy = unpacked_data[14:17]
        imu_data.temperature = unpacked_data[17]
        imu_data.timestamp = unpacked_data[18]
        imu_data.crc_32 = unpacked_data[19]

        return imu_data

    def parse_odometer_data(self, data: bytes) -> OdometerData:
        # 定义 odometer 数据帧格式
        odometer_format = 'B B H I 3f 4f 3f 3f I'
        # crc_format = 'I'

        # 从数据中提取数据体和 CRC32 校验码
        # data_body = data[:-4]  # 除去最后的 CRC32 校验码
        # 去除检验部分
        # received_crc = struct.unpack(crc_format, data[-4:])[0]  # 使用小端序解包 CRC32

        # computed_crc = crc32(data_body)
        # 调试输出
        # print(f"数据体: {data.hex()}")
        # print(f"接收到的 CRC32: {received_crc:08X}")
        # print(f"计算的 CRC32: {computed_crc:08X}")
        # if computed_crc != received_crc:
        #    raise ValueError(f"CRC32 校验失败: 计算值 {computed_crc:08X}, 接收到的值 {received_crc:08X}")

        # 解包数据体
        unpacked_data = struct.unpack(odometer_format, data)

        # 创建 OdometerData 实例
        odometer_data = OdometerData()
        odometer_data.header1 = unpacked_data[0]
        odometer_data.header2 = unpacked_data[1]
        odometer_data.sequence = unpacked_data[2]
        odometer_data.data_type = unpacked_data[3]
        odometer_data.position_x = unpacked_data[4]
        odometer_data.position_y = unpacked_data[5]
        odometer_data.position_z = unpacked_data[6]
        odometer_data.orientation_x = unpacked_data[7]
        odometer_data.orientation_y = unpacked_data[8]
        odometer_data.orientation_z = unpacked_data[9]
        odometer_data.orientation_w = unpacked_data[10]
        odometer_data.linear_x = unpacked_data[11]
        odometer_data.linear_y = unpacked_data[12]
        odometer_data.linear_z = unpacked_data[13]
        odometer_data.angular_x = unpacked_data[14]
        odometer_data.angular_y = unpacked_data[15]
        odometer_data.angular_z = unpacked_data[16]
        odometer_data.crc_32 = unpacked_data[17]

        return odometer_data

    def parse_motor_state_data(self, data: bytes, motor_count: int) -> MotorStateData:
        header_format = 'BBHI'
        motor_format = 'I3fi'
        crc_format = 'I'
        data_body = data[:-4]  # 除去最后的 CRC32 校验码
        received_crc = struct.unpack(crc_format, data[-4:])[0]  # 使用小端序解包 CRC32

        computed_crc = crc32(data_body)
        # 调试输出
        # print(f"数据体: {data.hex()}")
        # print(f"接收到的 CRC32: {received_crc:08X}")
        # print(f"计算的 CRC32: {computed_crc:08X}")

        if computed_crc != received_crc:
            raise ValueError(f"CRC32 校验失败: 计算值 {computed_crc:08X}, 接收到的值 {received_crc:08X}")
        motor_data = MotorStateData(motor_count)  # 创建 MotorStateData 实例，传入电机数量
        # 解析帧头部分
        header_data = struct.unpack(header_format, data[:8])
        motor_data.header1 = header_data[0]
        motor_data.header2 = header_data[1]
        motor_data.sequence = header_data[2]
        motor_data.data_type = header_data[3]
        # 解析电机状态部分
        offset = 8  # 帧头占用8字节
        for i in range(motor_count):
            motor_state_data = struct.unpack(motor_format, data[offset:offset + 20])
            motor_state = MotorState()  # 新建 MotorState 实例
            motor_state.mode = motor_state_data[0]
            motor_state.pos = motor_state_data[1]
            motor_state.w = motor_state_data[2]
            motor_state.t = motor_state_data[3]
            motor_state.temperature = motor_state_data[4]
            motor_data.state[i] = motor_state  # 将状态存储到对应位置
            offset += 20  # 每个电机状态占用20字节
        # 解析 CRC32
        motor_data.crc_32 = struct.unpack(crc_format, data[offset:offset + 4])[0]
        return motor_data

    def parse_joint_state_data(self, data: bytes, motor_count: int) -> JointStateData:
        header_format = 'BBHI'  # 帧头格式
        joint_format = '3f4f'  # 关节状态格式，3个 float 为位置，4个 float 为方向
        crc_format = 'I'  # CRC32 格式
        data_body = data[:-4]  # 除去最后的 CRC32 校验码
        received_crc = struct.unpack(crc_format, data[-4:])[0]  # 使用小端序解包 CRC32
        computed_crc = crc32(data_body)
        # 校验 CRC32
        if computed_crc != received_crc:
            raise ValueError(f"CRC32 校验失败: 计算值 {computed_crc:08X}, 接收到的值 {received_crc:08X}")
        joint_state_data = JointStateData(motor_count)  # 创建 JointStateData 实例，传入电机数量
        # 解析帧头部分
        header_data = struct.unpack(header_format, data[:8])
        joint_state_data.header1 = header_data[0]
        joint_state_data.header2 = header_data[1]
        joint_state_data.sequence = header_data[2]
        joint_state_data.data_type = header_data[3]
        # 解析关节状态部分
        offset = 8  # 帧头占用8字节
        for i in range(motor_count):
            joint_data = struct.unpack(joint_format, data[offset:offset + 28])  # 每个关节状态占用28字节
            joint_state = JointState()  # 新建 JointState 实例

            # 解析位置
            joint_state.position = JointPosition(x=joint_data[0], y=joint_data[1], z=joint_data[2])
            # 解析方向
            joint_state.orientation = JointOrientation(w=joint_data[3], x=joint_data[4], y=joint_data[5],
                                                       z=joint_data[6])

            joint_state_data.state[i] = joint_state  # 将状态存储到对应位置
            offset += 28  # 每个关节状态占用28字节
        # 解析 CRC32
        joint_state_data.crc_32 = struct.unpack(crc_format, data[offset:offset + 4])[0]
        return joint_state_data

    def parse_remote_control_data(self, data: bytes) -> RemoteData:
        # 解析格式字符串 BBHIH6B4fI
        remote_control_format = 'BBHIH6B4fI'
        crc_format = 'I'

        data_body = data[:-4]  # 除去最后的 CRC32 校验码
        received_crc = struct.unpack(crc_format, data[-4:])[0]  # 使用小端序解包 CRC32

        computed_crc = crc32(data_body)
        if computed_crc != received_crc:
            raise ValueError(f"CRC32 校验失败: 计算值 {computed_crc:08X}, 接收到的值 {received_crc:08X}")
        # 按照格式解析传入的字节数据
        unpacked_data = struct.unpack(remote_control_format, data)
        remote_data = RemoteData()
        # 解包结果
        remote_data.header1 = unpacked_data[0]
        remote_data.header2 = unpacked_data[1]
        remote_data.sequence = unpacked_data[2]
        remote_data.data_type = unpacked_data[3]
        remote_data.remote_type = unpacked_data[4]
        # 解析按键信息（6 个按钮字段）
        remote_data.swa = unpacked_data[5]
        remote_data.swb = unpacked_data[6]
        remote_data.swe = unpacked_data[7]
        remote_data.swg = unpacked_data[8]
        remote_data.reserved1 = unpacked_data[9]
        remote_data.reserved2 = unpacked_data[10]
        # 解析摇杆信息
        remote_data.rocker_left_x = unpacked_data[11]
        remote_data.rocker_left_y = unpacked_data[12]
        remote_data.rocker_right_x = unpacked_data[13]
        remote_data.rocker_right_y = unpacked_data[14]
        # CRC 校验码
        remote_data.crc_32 = unpacked_data[15]

        return remote_data  # 返回解析后的数据对象

    # 机器人数据接收线程函数
    def receive_and_process_data(self):
        while True:
            data, _ = self.sock.recvfrom(1024)  # 假设接收到的数据长度不会超过1024字节

            if len(data) == 72:  # IMU_DATA 的数据长度为72字节
                try:
                    imu_data = self.parse_imu_data(data)
                    with self.imu_data_lock:
                        self.imu_data = imu_data
                except ValueError as e:
                    print(f"IMU 数据校验失败: {e}")
            elif len(data) == 252:  # MOTOR_STATE_DATA 的数据长度为252字节
                try:
                    motor_state_data = self.parse_motor_state_data(data, motor_count=12)  # 电机数量12
                    with self.motor_data_lock:
                        self.motor_state_data = motor_state_data
                except ValueError as e:
                    print(f"Motor 状态数据校验失败: {e}")
            elif len(data) == 172:  # MOTOR_STATE_DATA(手臂) 的数据长度为172字节
                try:
                    arm_state_data = self.parse_motor_state_data(data, motor_count=8)  # 手臂电机数量8
                    arm_type = arm_state_data.data_type
                    with self.arm_data_lock:
                        if arm_type == 0x08:  # 左手臂数据
                            self.left_arm_state_data = arm_state_data
                        elif arm_type == 0x09:  # 右手臂数据
                            self.right_arm_state_data = arm_state_data
                except ValueError as e:
                    print(f"Motor 状态数据校验失败: {e}")

            elif len(data) == 64:  # 里程计数据长度为64字节
                try:
                    odometer_data = self.parse_odometer_data(data)
                    with self.odometer_data_lock:
                        self.odometer_data = odometer_data
                except ValueError as e:
                    print(f"Odo 状态数据校验失败: {e}")
            elif len(data) == 908:
                try:
                    joint_state_data = self.parse_joint_state_data(data, motor_count=32)
                    data_type = joint_state_data.data_type
                    with self.joint_state_data_lock:
                        if data_type == 0x0c:  # 位姿数据
                            self.joint_state_data = joint_state_data
                except ValueError as e:
                    print(f"Motor 位姿数据校验失败: {e}")

            elif len(data) == 36:  # remote_data 数据长度为36字节
                try:
                    remote_data = self.parse_remote_control_data(data)
                    with self.remote_data_lock:
                        self.remote_data = remote_data
                except ValueError as e:
                    print(f"Remote 遥控器数据校验失败: {e}")

    # IMU数据获取
    def get_imu_data(self):
        with self.imu_data_lock:
            return self.imu_data

    # 里程计数据获取
    def get_odometer_data(self):
        with self.odometer_data_lock:
            return self.odometer_data

    # 电机数据获取
    def get_motor_state_data(self):
        with self.motor_data_lock:
            return self.motor_state_data

    # 左臂电机数据获取
    def get_left_arm_state_data(self):
        with self.arm_data_lock:
            return self.left_arm_state_data

    # 右臂电机数据获取
    def get_right_arm_state_data(self):
        with self.arm_data_lock:
            return self.right_arm_state_data

    # 关节位姿数据获取
    def get_joint_state_data(self):
        with self.joint_state_data_lock:
            return self.joint_state_data

    # 遥控器数据获取
    def get_remote_data(self):
        with self.remote_data_lock:
            return self.remote_data


# 电机命令类
class MotorCmd:
    def __init__(self, mode=0, pos=0.0, w=0.0, t=0.0, kp=0.0, kd=0.0):
        self.mode = mode  # 电机模式，默认不要控制
        self.pos = pos  # 电机位置，单位为rad
        self.w = w  # 电机角速度，单位为rad/s
        self.t = t  # 前馈力矩
        self.kp = kp  # 位置刚度系数
        self.kd = kd  # 角速度刚度系数

    def __repr__(self):
        return (f"MotorCmd(mode={self.mode}, pos={self.pos}, w={self.w}, "
                f"t={self.t}, kp={self.kp}, kd={self.kd})")


class MotorCmdDataHandler:
    def __init__(self, num_motors, header1=0, header2=0, sequence=0, data_type=0, crc_32=0):
        self.header1 = header1
        self.header2 = header2
        self.sequence = sequence
        self.data_type = data_type
        self.cmd = [MotorCmd() for _ in range(num_motors)]  # 创建一个包含num_motors个MotorCmd实例的列表
        self.crc_32 = crc_32
        self.sock = None

    def __repr__(self):
        cmd_repr = ', '.join(repr(cmd) for cmd in self.cmd)
        return (f"MotorCmdData(header1={self.header1}, header2={self.header2}, "
                f"sequence={self.sequence}, data_type={self.data_type}, "
                f"cmd=[{cmd_repr}], crc_32={self.crc_32})")

    def pack_data(self) -> bytes:
        # 序列号自加
        if self.sequence < 65535:
            self.sequence += 1
        else:
            self.sequence = 0
            # 先打包头部
        header_format = 'BBHI'  # uint8_t header1, uint8_t header2, uint16_t sequence, uint32_t data_type
        header_data = struct.pack(header_format, self.header1, self.header2, self.sequence, self.data_type)

        # 打包MOTOR_CMD部分
        cmd_format = 'If4f'  # uint32_t mode, float pos, float w, float t, float kp, float kd
        # 使用实际的电机数量
        cmd_data = b''.join(
            struct.pack(cmd_format, cmd.mode, cmd.pos, cmd.w, cmd.t, cmd.kp, cmd.kd)
            for cmd in self.cmd
        )
        # 组合数据并计算CRC
        combined_data = header_data + cmd_data
        self.crc_32 = crc32(combined_data)  # 计算 CRC 值

        # 打包crc_32部分
        crc_format = 'I'  # uint32_t crc_32
        crc_data = struct.pack(crc_format, self.crc_32)

        # 最后将所有部分组合在一起
        full_data = combined_data + crc_data
        return full_data

    def send_data(self):
        # 如果socket还没有初始化，初始化一个UDP socket
        if self.sock is None:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 打包数据
        packed_data = self.pack_data()

        # 通过socket发送数据
        self.sock.sendto(packed_data, ("192.168.93.1", 6061))
        return

    def close_socket(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None


# 电机状态数据打印到Windaow上位机
class MotorStateDataSendtoWindows:
    def __init__(self, dest_ip='192.168.93.66', dest_port=10001, local_port=10002):
        # 创建一个UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 绑定本地端口
        self.sock.bind(('', local_port))
        self.dest_ip = dest_ip
        self.dest_port = dest_port

    # 打印顺序：电机位置指令信息, 腿部电机位置、速度和扭矩信息, imu欧拉角(rpy)四元数(wxyz), 里程计位置信息(xyz)
    def send_robot_state_data(self, motor_cmd_data: MotorCmdDataHandler, leg_motor_data: MotorStateData,
                              imu_data: ImuData, odo_data: OdometerData):
        # 数据包组装
        data = b''
        # 下发的位置指令值
        for motor_cmd in motor_cmd_data.cmd:
            # 按float类型组装pos, w, t
            data += struct.pack('f', motor_cmd.pos)
        # 腿部电机的位置值、速度值、扭矩值
        for leg_motor in leg_motor_data.state:
            # 按float类型组装pos, w, t
            data += struct.pack('fff', leg_motor.pos, leg_motor.w, leg_motor.t)

        # imu欧拉角(rpy)和四元数(wxyz)
        data += struct.pack('fff', imu_data.rpy[0], imu_data.rpy[1], imu_data.rpy[2])
        data += struct.pack('ffff', imu_data.quaternion[0], imu_data.quaternion[1], imu_data.quaternion[2],
                            imu_data.quaternion[3])
        # 里程计position_x、y、z
        data += struct.pack('fff', odo_data.position_x, odo_data.position_y, odo_data.position_z)
        # 添加帧尾的固定4字节
        data += b'\x00\x00\x80\x7f'
        # 发送数据包
        self.sock.sendto(data, (self.dest_ip, self.dest_port))

    def send_motor_data(self, leg_motor_data: MotorStateData):
        # 数据包组装
        data = b''
        # 腿部电机的位置值、速度值、扭矩值
        for leg_motor in leg_motor_data.state:
            # 按float类型组装pos, w, t
            data += struct.pack('fff', leg_motor.pos, leg_motor.w, leg_motor.t)
        # 添加帧尾的固定4字节
        data += b'\x00\x00\x80\x7f'
        # 发送数据包
        self.sock.sendto(data, (self.dest_ip, self.dest_port))

    def send_remote_data(self, remote_data: RemoteData):
        # 数据包组装
        data = b''
        # 按键值
        data += struct.pack('ffff', remote_data.swa, remote_data.swb, remote_data.swe, remote_data.swg)
        # 摇杆值
        data += struct.pack('ffff', remote_data.rocker_left_x, remote_data.rocker_left_y, remote_data.rocker_right_x,
                            remote_data.rocker_right_y)
        # 添加帧尾的固定4字节
        data += b'\x00\x00\x80\x7f'
        # 发送数据包
        self.sock.sendto(data, (self.dest_ip, self.dest_port))

    def send_imu_data(self, imu_data: ImuData):
        # 数据包组装
        data = b''
        # imu欧拉角(rpy)和四元数(wxyz)
        data += struct.pack('fff', imu_data.rpy[0], imu_data.rpy[1], imu_data.rpy[2])
        data += struct.pack('ffff', imu_data.quaternion[0], imu_data.quaternion[1], imu_data.quaternion[2],
                            imu_data.quaternion[3])
        # 添加帧尾的固定4字节
        data += b'\x00\x00\x80\x7f'
        # 发送数据包
        self.sock.sendto(data, (self.dest_ip, self.dest_port))

    def close(self):
        # 关闭socket
        self.sock.close()