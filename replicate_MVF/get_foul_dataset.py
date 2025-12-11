from SoccerNet.Downloader import SoccerNetDownloader as SNdl
from utils import load_env_vars
import os

load_env_vars()

mySNdl = SNdl(LocalDirectory=os.environ.get("SOCCER_DATA", ""))
mySNdl.downloadDataTask(task="mvfouls", split=["train","valid","test","challenge"], password="s0cc3rn3t")


