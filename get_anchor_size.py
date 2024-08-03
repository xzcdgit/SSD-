import numpy as np

def generate_anchors(image_size, feature_map_sizes, min_sizes, max_sizes, aspect_ratios):
    anchors = []
    for k, f in enumerate(feature_map_sizes):
        for i in range(f):
            for j in range(f):
                f_k = image_size / f
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = min_sizes[k] / image_size
                anchors.append([cx, cy, s_k, s_k])

                s_k_prime = np.sqrt(s_k * (max_sizes[k] / image_size))
                anchors.append([cx, cy, s_k_prime, s_k_prime])

                for ar in aspect_ratios[k]:
                    anchors.append([cx, cy, s_k * np.sqrt(ar), s_k / np.sqrt(ar)])
                    anchors.append([cx, cy, s_k / np.sqrt(ar), s_k * np.sqrt(ar)])
    return np.array(anchors)

image_size = 600
feature_map_sizes = [38, 19, 10, 5, 3, 1]
min_sizes = [30, 60, 111, 162, 213, 264]
max_sizes = [60, 111, 162, 213, 264, 315]
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

anchors = generate_anchors(image_size, feature_map_sizes, min_sizes, max_sizes, aspect_ratios)
print(anchors)
