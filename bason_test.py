
import bson

def load_episode_data(filename: str) -> bool:
    """加载episode_0.bson文件"""
    
    try:
        with open(filename, 'rb') as f:
            data = f.read()
            episode_data = bson.decode(data)
        import pdb; pdb.set_trace()
    except:
        pass


load_episode_data('/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT/data/ours/all/action4/episode_0/episode_0.bson')