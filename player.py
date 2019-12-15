import psycopg2
import settings_local as SETTINGS
import time
from PIL import Image
import numpy as np
from kmeans import kmeans
import requests

def rgb2h(img):
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    h = np.zeros((img.shape[0], img.shape[1]))

    MAX = img.max(axis=2)
    MIN = img.min(axis=2)

    var0 = (MAX == MIN)
    h[var0] = 0
    var1 = (MAX != MIN) & (MAX == r) & (g < b)
    h[var1] = 60 * (g[var1] - b[var1]) / (MAX[var1] - MIN[var1]) + 360
    var2 = (MAX != MIN) & (MAX == r) & (g >= b)
    h[var2] = 60 * (g[var2] - b[var2]) / (MAX[var2] - MIN[var2]) + 0
    var3 = (MAX != MIN) & (MAX == g)
    h[var3] = 60 * (b[var3] - r[var3]) / (MAX[var3] - MIN[var3]) + 120
    var4 = (MAX != MIN) & (MAX == b)
    h[var4] = 60 * (r[var4] - g[var4]) / (MAX[var4] - MIN[var4]) + 240

    return h


def cluster_labels2img(labels, arr):
    var0 = (labels == 0)
    var1 = (labels == 1)
    var2 = (labels == 2)

    arr[var0] = [255, 0, 0]
    arr[var1] = [0, 255, 0]
    arr[var2] = [0, 0, 255]

    return arr

print('hey', flush=True)
conn = psycopg2.connect(host=SETTINGS.host, dbname=SETTINGS.name, user=SETTINGS.user, password=SETTINGS.password)
cur = conn.cursor()
print('hey', flush=True)
cur.execute("SELECT id, link FROM worker_task WHERE status=%s", (False,))
tasks = cur.fetchall()
for task in tasks:

    image_response = requests.get(task[1])
    with open('image.jpg', 'wb') as f:
        # print(image_response.content)
        f.write(image_response.content)

    image = Image.open('image.jpg')
    arr = np.array(image)
    m = arr.shape[0]
    n = arr.shape[1]
    h = rgb2h(arr)
    h = np.reshape(h, (m * n, 1))

    labels = kmeans(h, 3, 0.1)

    arr = np.reshape(arr, (m * n, 3))

    arr = cluster_labels2img(labels, arr)

    arr = np.reshape(arr, (m, n, 3))

    img = Image.fromarray(arr, 'RGB')
    img.save('new.jpg')
    with open('new.jpg', 'rb') as f:
        image_bytes = f.read()
        cur.execute("UPDATE worker_task SET status=%s, bytes=%s WHERE id=%s", (True, image_bytes, task[0]))
        conn.commit()

print('hey', flush=True)


