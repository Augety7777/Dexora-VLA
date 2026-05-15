import cv2
import os
import glob
from natsort import natsorted

def create_video_from_images(image_folder, output_video, fps=20):
    """
    将文件夹中的图片组合成视频
    
    参数:
        image_folder (str): 包含图片的文件夹路径
        output_video (str): 输出视频文件路径
        fps (int): 帧率，默认为20
    """
    # 获取所有图片文件（支持常见格式）
    image_files = glob.glob(os.path.join(image_folder, '*.[jJ][pP][gG]')) + \
                  glob.glob(os.path.join(image_folder, '*.[pP][nN][gG]')) + \
                  glob.glob(os.path.join(image_folder, '*.[jJ][pP][eE][gG]')) + \
                  glob.glob(os.path.join(image_folder, '*.[bB][mM][pP]')) + \
                  glob.glob(os.path.join(image_folder, '*.[tT][iI][fF][fF]'))
    
    # 使用自然排序（按数字顺序而不是字母顺序）
    image_files = natsorted(image_files)
    
    if not image_files:
        print(f"在文件夹 {image_folder} 中没有找到图片文件")
        return
    
    # 读取第一张图片获取尺寸
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape
    
    # 定义视频编码器（根据文件扩展名自动选择）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 对于.mp4文件
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    print(f"开始创建视频，共 {len(image_files)} 张图片...")
    
    for i, image_file in enumerate(image_files):
        img = cv2.imread(image_file)
        if img is None:
            print(f"警告: 无法读取图片 {image_file}，跳过")
            continue
            
        # 确保图片尺寸一致
        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))
            
        video.write(img)
        
        # 打印进度
        if (i+1) % 10 == 0 or (i+1) == len(image_files):
            print(f"已处理 {i+1}/{len(image_files)} 张图片")
    
    video.release()
    print(f"视频已保存到 {output_video}")

if __name__ == "__main__":
    # 输入参数
    image_folder = "/home/air/Desktop/wzr/data_collection/action6/episode_29/camera_2"

    output_video = "/home/air/Desktop/wzr/data_collection/action6/episode_29/camera_2.mp4"

    # 检查文件夹是否存在
    if not os.path.isdir(image_folder):
        print(f"错误: 文件夹 {image_folder} 不存在")
    else:
        create_video_from_images(image_folder, output_video, fps=20)