from bson import BSON
import argparse
from pprint import pprint
from typing import Any
import av
from io import BytesIO


def decode_h264(h264_bytes: bytes) -> Any:
    inbuf = BytesIO(h264_bytes)
    container = av.open(inbuf)
    ret = [
        {
            "t": int(frame.pts * frame.time_base * 1e3),
            "data": frame.to_ndarray(format="bgr24"),
        }
        for frame in container.decode(video=0)
    ]
    assert len(ret) > 0, "No frames found in h264"
    return ret


def save_bson(bson_file: str, data: dict):
    with open(bson_file, "wb") as f:
        f.write(BSON.encode(data))
    print(f"Saved BSON data to {bson_file}")


def load_bson(bson_file: str) -> dict:
    with open(bson_file, "rb") as f:
        data = BSON.decode(f.read())
    print(f"Loaded BSON data from {bson_file}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Show BSON file content")
    parser.add_argument("bson_file", type=str, help="Path to the BSON file")
    parser.add_argument(
        "-s",
        "--start-index",
        type=int,
        default=0,
        help="Start index for displaying data",
    )
    parser.add_argument(
        "-e",
        "--end-index",
        type=int,
        default=10,
        help="End index for displaying data",
    )
    args = parser.parse_args()

    all_images = {}

    data = load_bson(args.bson_file)
    print(data.keys())
    print(data["data"].keys())
    pprint(data["metadata"]["topics"])

    for i in range(args.start_index, args.end_index):
        print(f"Index: {i}")
        for topic in data["data"].keys():
            if "image" in topic:
                if topic not in all_images:
                    img_data = data["data"][topic]
                    if isinstance(img_data, bytes):
                        all_images[topic] = decode_h264(img_data)
                    else:
                        all_images[topic] = img_data
                images = all_images[topic]
                image = images[i]["data"]
                image_t = images[i]["t"]
                print(topic, image.shape, image.dtype, image_t)
            else:
                print(topic, data["data"][topic][i])
        print("\n")


if __name__ == "__main__":
    main()
