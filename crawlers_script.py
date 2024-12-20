import requests
from PIL import Image
from io import BytesIO
import pandas as pd
import os
import time
from tqdm import tqdm
import numpy as np

def get_metadata(id, request_type):
    if request_type == "litepoints":
        url = f"https://api.anitabi.cn/bangumi/{id}/lite"
    elif request_type == "anime":
        url = f"https://api.anitabi.cn/bangumi/{id}"
    elif request_type == "points_detail":
        url = f"https://api.anitabi.cn/bangumi/{id}/points/detail"
    response = requests.get(url)
    if response.status_code == 200:
        # 请求成功，打印返回的 JSON 数据
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None    

def get_images(img_url):
    response = requests.get(img_url)
    if response.status_code == 200:
        # 请求成功，将图像数据转换为 PIL 图像对象
        image = Image.open(BytesIO(response.content))
        # 显示图像
        return image
    else:
        print(f"Error: {response.status_code}")
        return None


def crawl(anime_id):
    metadata_savepath = "/data_nas/cehou/anitabi/metadata/"
    os.makedirs(os.path.join(metadata_savepath, f"{anime_id}"), exist_ok=True)
    print(f"Start crawling metadata for anime with ID: {anime_id}")
    litepoints_data = get_metadata(anime_id, "litepoints")
    litepoints_df = pd.DataFrame(litepoints_data['litePoints'])
    litepoints_df.to_csv(os.path.join(metadata_savepath, f"{anime_id}/litepoints_metadata.csv"), index=False)

    anime_data = get_metadata(anime_id, "anime")
    anime_df = pd.DataFrame([anime_data])
    anime_df.to_csv(os.path.join(metadata_savepath, f"{anime_id}/anime_metadata.csv"), index=False)
    print(f"Bangumi title: {anime_data['title']}")

    detailed_points_data = get_metadata(anime_id, "points_detail")
    detailed_points_df = pd.DataFrame(detailed_points_data)
    print(f"Total detailed points: {len(detailed_points_df)}")
    detailed_points_df.to_csv(os.path.join(metadata_savepath, f"{anime_id}/detailed_points_metadata.csv"), index=False)

    # 爬取图像
    if 'image' not in detailed_points_df.columns:
        print("No image column in detailed points metadata, skip crawling images")
        return
    
    anime_image_folder = os.path.join(metadata_savepath, f"{anime_id}/anime_imgs")
    os.makedirs(anime_image_folder, exist_ok=True)
    for i, row in tqdm(detailed_points_df.iterrows()):

        if row.image is np.nan:
            continue
        else:
            img_url = row.image.split('.jpg')[0] + '.jpg'

        anime_img_id = row.id
        anime_image = get_images(img_url)
        try:
            anime_image.save(os.path.join(anime_image_folder, f"{anime_img_id}.jpg"))
        except Exception as e:
            print(f"Error saving image with ID: {anime_img_id}, cannot write mode RGBA as JPEG")
        time.sleep(0.1)


if __name__ == "__main__":
    anitabi_dir = "/data_nas/cehou/anitabi/metadata/"
    exists = [int(i) for i in os.listdir(anitabi_dir)]
    for i in tqdm(range(1, 5)):
        if i in exists:
            continue
        else:
            try:
                crawl(i)
            except Exception as e:
                print(f"Error crawling anime with ID: {i}, {e}")
                continue