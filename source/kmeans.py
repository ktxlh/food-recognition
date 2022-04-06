from __future__ import absolute_import, annotations
from __future__ import print_function
from __future__ import division
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import json

def image_to_matrix(image_file, grays=False):
    img = plt.imread(image_file)
    # in case of transparency values
    if len(img.shape) == 3 and img.shape[2] > 3:
        height, width, depth = img.shape
        new_img = np.zeros([height, width, 3])
        for r in range(height):
            for c in range(width):
                new_img[r, c, :] = img[r, c, 0:3]
        img = np.copy(new_img)
    if grays and len(img.shape) == 3:
        height, width = img.shape[0:2]
        new_img = np.zeros([height, width])
        for r in range(height):
            for c in range(width):
                new_img[r, c] = img[r, c, 0]
        img = new_img
    return img
def image_compression(image_values, p):
    reconstimg = np.zeros(image_values.shape)
    r, g, b = image_values[:,:,0], image_values[:,:,1], image_values[:,:,2]
    
    U, sigma, V = np.linalg.svd(r)
    reconstimg[:,:,0] = np.matrix(U[:, :p]) * np.diag(sigma[:p]) * np.matrix(V[:p, :])
    U, sigma, V = np.linalg.svd(g)
    reconstimg[:,:,1] = np.matrix(U[:, :p]) * np.diag(sigma[:p]) * np.matrix(V[:p, :])
    U, sigma, V = np.linalg.svd(b)
    reconstimg[:,:,2] = np.matrix(U[:, :p]) * np.diag(sigma[:p]) * np.matrix(V[:p, :])
    
    for ind1, row in enumerate(reconstimg):
        for ind2, col in enumerate(row):
            for ind3, value in enumerate(col):
                if value < 0:
                    reconstimg[ind1,ind2,ind3] = abs(value)
                if value > 255:
                    reconstimg[ind1,ind2,ind3] = 255
    reconstimg = reconstimg.astype(np.uint8)
    print (reconstimg.shape)
    return reconstimg
def update_image_values(image_values, k):
    r = image_values.shape[0]
    c = image_values.shape[1]
    ch = image_values.shape[2]
    image_values = image_values.reshape(r*c,ch)
    
    kmeans = KMeans(n_clusters=k, random_state=0).fit(image_values)
    labels, centers = kmeans.labels_, kmeans.cluster_centers_
    updated_image_values = np.copy(image_values)
    calinski_harabasz_score = metrics.calinski_harabasz_score(image_values, labels)/(r*c)
    print ('calinski_harabasz_score: %d' % calinski_harabasz_score)
    #silhouette_score = metrics.silhouette_score(image_values, labels, metric='euclidean')
    davies_bouldin_score = metrics.davies_bouldin_score(image_values, labels)
    
    #print ('silhouette_score: %d' % silhouette_score)
    print ('davies_bouldin_score: %d' % davies_bouldin_score)
    for i in range(0, k):
        indices_current_cluster = np.where(labels == i)[0]
        updated_image_values[indices_current_cluster] = centers[i]
    updated_image_values = updated_image_values.reshape(r,c,ch)
    return updated_image_values

def plot_image(img_list, title_list, figsize=(9, 12)):
    plt.figure(dpi=120)
    fig, axes = plt.subplots(1, len(img_list), figsize=figsize)
    for i, ax in enumerate(axes):
        ax.imshow(img_list[i])
        ax.set_title(title_list[i])
        ax.axis('off')
    
def load_picture(path):
    return image_to_matrix(path)
def calculate_per_image(path, p=10 , k=4):
    image_values = load_picture(path)
    image_svd = image_compression(image_values, p)
    updated_image_values = update_image_values(image_values, k)
    updated_image_values_with_pca = update_image_values(image_svd, k)
    plot_image([image_values, image_svd, updated_image_values, updated_image_values_with_pca], \
               ['Orignal', 'P = ' + str(p), "K = " + str(k), 'P = ' + str(p) + ", K = " + str(k)]  )
    plt.savefig(path.split('.j')[0] + '_kmean.png', dpi=120, bbox_inches='tight')
def find_optimal_num_clusters(path, max_P= 10, max_K=15): 
    image_values = load_picture(path)
    r = image_values.shape[0]
    c = image_values.shape[1]
    ch = image_values.shape[2]
    image_svd = image_compression(image_values, max_P)
    data = image_svd.reshape(r*c,ch)

    np.random.seed(1)
    losses = []
    ch_scores = []
    for k in range(2, max_K+1):
       # cluster_idx, centers, loss = KMeans()(data, k)
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        print (k)
        labels, centers, loss = kmeans.labels_, kmeans.cluster_centers_, kmeans.inertia_
        calinski_harabasz_score = metrics.calinski_harabasz_score(data, labels)/(r*c)
        ch_scores.append(calinski_harabasz_score)
        #il_coeff = silhouette_score(data, labels, metric='euclidean')
        #il_coeffes.append(il_coeff)
        losses.append(loss)
    
    return losses, ch_scores
def draw_elbow_figure(losses, max_K=15):
    x = [i for i in range(2,max_K+1)]
    plt.figure(dpi=600)
    for loss in losses:
        plt.plot(x, loss)
    plt.xlabel('k')
    plt.ylabel('loss')
    plt.title("Elbow method for different figures")
    #plt.plot(il_coeffes)
def draw_calinski_harabasz_score_figure(ch_scores, max_K=15):
    x = [i for i in range(2,max_K+1)]
    plt.figure(dpi=600)
    for score in ch_scores:
        plt.plot(x, score)
    plt.xlabel('k')
    plt.ylabel('normalized Calinski Harabasz score')
    plt.title("Calinski Harabasz score for different figures")
'''
plt.clf()
calculate_per_image('../131323.jpg')
calculate_per_image('../174956.jpg', 10, 8)
calculate_per_image('../131094.jpg', 10, 8)
'''

loss1, ch_scores1 = find_optimal_num_clusters('../131323.jpg')
loss2, ch_scores2 = find_optimal_num_clusters('../174956.jpg')
loss3, ch_scores3 = find_optimal_num_clusters('../131094.jpg')
draw_elbow_figure([loss1, loss2, loss3])
draw_calinski_harabasz_score_figure([ch_scores1, ch_scores2, ch_scores3])

'''
find_optimal_num_clusters('./images/008675.jpg')
find_optimal_num_clusters('./images/008827.jpg')
find_optimal_num_clusters('./images/008964.jpg')
'''
#f = open('annotations.json')
#data = json.load(f)
#print (data.keys())
#target = '011275'
#for k, v in data.items():
#    if v['id'] == target:
#        print (k)
#f.close()
#pic = {"id": 11275, "file_name": "011275.jpg", "width": 480, "height": 480}
#annotations = {"id": 9829, "image_id": 11275, "category_id": 101266, "segmentation": [[182.0, 419.5, 155.0, 419.5, 154.0, 418.5, 142.0, 418.5, 141.0, 417.5, 136.0, 417.5, 135.0, 416.5, 132.0, 416.5, 131.0, 415.5, 129.0, 415.5, 128.0, 414.5, 125.00000000000001, 414.5, 124.00000000000001, 413.5, 121.0, 413.5, 120.0, 412.5, 117.0, 412.5, 116.0, 411.5, 111.0, 411.5, 110.0, 410.5, 105.0, 410.5, 104.0, 409.5, 100.0, 409.5, 99.0, 408.5, 95.0, 408.5, 94.0, 407.5, 92.0, 407.5, 91.0, 406.5, 89.0, 406.5, 87.0, 404.5, 86.0, 404.5, 85.0, 403.5, 84.0, 403.5, 83.0, 402.5, 82.0, 402.5, 81.0, 401.5, 79.0, 401.5, 78.0, 400.5, 76.0, 400.5, 75.0, 399.5, 74.0, 399.5, 73.0, 398.5, 72.0, 398.5, 71.0, 397.5, 70.0, 397.5, 69.0, 396.5, 68.0, 396.5, 65.0, 393.5, 64.0, 393.5, 63.0, 392.5, 62.00000000000001, 392.5, 61.0, 391.5, 60.0, 391.5, 59.0, 390.5, 58.0, 390.5, 56.0, 388.5, 55.0, 388.5, 45.0, 378.5, 44.0, 378.5, 41.0, 375.5, 40.0, 375.5, 39.5, 375.0, 39.5, 374.0, 36.5, 371.0, 36.5, 370.0, 35.5, 369.0, 35.5, 368.0, 34.5, 367.0, 34.5, 366.0, 33.5, 365.0, 33.5, 364.0, 32.5, 363.0, 32.5, 362.0, 31.5, 361.0, 31.5, 359.0, 30.5, 358.0, 30.5, 357.0, 29.5, 356.0, 29.5, 355.0, 28.5, 354.0, 28.5, 353.0, 27.5, 352.0, 27.5, 351.0, 26.5, 350.0, 26.5, 348.0, 25.5, 347.0, 25.5, 346.0, 24.5, 345.0, 24.5, 336.0, 23.5, 335.0, 23.5, 327.0, 22.5, 326.0, 22.5, 320.0, 21.5, 319.0, 21.5, 313.0, 20.5, 312.0, 20.5, 307.0, 19.5, 306.0, 19.5, 300.0, 18.5, 299.0, 18.5, 290.0, 17.5, 289.0, 17.5, 280.0, 18.5, 279.0, 18.5, 275.0, 19.5, 274.0, 19.5, 271.0, 20.5, 270.0, 20.5, 269.0, 21.5, 268.0, 21.5, 266.0, 22.5, 265.0, 22.5, 264.0, 23.5, 263.0, 23.5, 241.0, 22.5, 240.0, 22.5, 239.0, 21.5, 238.0, 21.5, 237.0, 20.5, 236.0, 20.5, 234.0, 19.5, 233.0, 19.5, 221.0, 20.5, 220.0, 20.5, 219.0, 27.0, 212.5, 28.0, 213.5, 29.0, 213.5, 30.0, 214.5, 32.0, 214.5, 33.0, 215.5, 35.0, 215.5, 36.0, 216.5, 39.0, 216.5, 40.0, 217.5, 46.0, 217.5, 47.0, 218.5, 70.0, 218.5, 71.0, 219.5, 119.0, 219.5, 120.0, 220.5, 135.0, 220.5, 136.0, 219.5, 145.0, 219.5, 146.0, 220.5, 148.0, 220.5, 149.5, 222.0, 148.0, 223.5, 147.0, 223.5, 144.0, 226.5, 143.0, 226.5, 141.0, 228.5, 137.0, 228.5, 136.0, 229.5, 133.0, 229.5, 132.0, 230.5, 130.0, 230.5, 129.5, 231.0, 130.0, 231.5, 132.0, 231.5, 133.0, 232.5, 152.0, 232.5, 153.0, 233.5, 158.0, 233.5, 159.0, 232.5, 165.0, 232.5, 166.0, 231.5, 168.0, 231.5, 168.5, 231.0, 168.5, 222.0, 168.0, 221.5, 167.0, 221.5, 166.0, 220.5, 163.0, 220.5, 162.0, 219.5, 161.0, 219.5, 159.5, 218.0, 159.5, 217.0, 158.5, 216.0, 158.5, 214.0, 157.5, 213.0, 157.5, 211.0, 158.5, 210.0, 158.5, 209.0, 160.5, 207.0, 160.5, 206.0, 161.5, 205.0, 161.5, 204.0, 162.5, 203.0, 162.5, 202.0, 163.5, 201.0, 163.5, 197.0, 164.5, 196.0, 164.5, 194.0, 165.5, 193.0, 165.5, 191.0, 169.0, 187.5, 170.0, 187.5, 172.0, 185.5, 173.0, 185.5, 174.0, 184.5, 175.0, 184.5, 177.0, 182.5, 178.0, 182.5, 180.0, 180.5, 181.0, 180.5, 183.0, 178.5, 184.0, 178.5, 185.0, 177.5, 186.0, 177.5, 187.0, 176.5, 188.0, 176.5, 189.0, 175.5, 190.0, 175.5, 191.0, 174.5, 192.0, 174.5, 193.0, 173.5, 194.0, 173.5, 198.0, 169.5, 199.0, 169.5, 201.0, 167.5, 202.0, 167.5, 203.0, 166.5, 204.0, 166.5, 205.0, 165.5, 206.0, 165.5, 207.0, 164.5, 211.0, 164.5, 212.0, 163.5, 215.0, 163.5, 216.0, 162.5, 219.0, 162.5, 220.0, 161.5, 224.0, 161.5, 225.0, 160.5, 229.0, 160.5, 230.0, 159.5, 235.0, 159.5, 236.0, 158.5, 261.0, 158.5, 262.0, 159.5, 268.0, 159.5, 269.0, 160.5, 273.0, 160.5, 274.0, 161.5, 280.0, 161.5, 281.0, 162.5, 287.0, 162.5, 288.0, 163.5, 291.0, 163.5, 292.0, 164.5, 294.0, 164.5, 295.0, 165.5, 297.0, 165.5, 298.0, 166.5, 300.0, 166.5, 301.0, 167.5, 302.0, 167.5, 303.0, 168.5, 306.0, 168.5, 307.0, 169.5, 313.0, 169.5, 314.0, 170.5, 320.0, 170.5, 321.0, 171.5, 323.0, 171.5, 324.0, 172.5, 327.0, 172.5, 328.0, 173.5, 332.0, 173.5, 333.0, 174.5, 337.0, 174.5, 338.0, 175.5, 350.0, 175.5, 351.0, 176.5, 355.0, 176.5, 356.0, 177.5, 357.0, 177.5, 358.0, 178.5, 360.0, 178.5, 361.0, 179.5, 362.0, 179.5, 363.0, 180.5, 364.0, 180.5, 366.0, 182.5, 367.0, 182.5, 368.0, 183.5, 369.0, 183.5, 377.0, 191.5, 378.0, 191.5, 380.0, 193.5, 381.0, 193.5, 382.0, 194.5, 384.0, 194.5, 385.0, 195.5, 386.0, 195.5, 387.0, 196.5, 388.0, 196.5, 389.0, 197.5, 390.0, 197.5, 392.0, 199.5, 393.0, 199.5, 397.0, 203.5, 398.0, 203.5, 399.0, 204.5, 400.0, 204.5, 401.0, 205.5, 402.0, 205.5, 403.0, 206.5, 404.0, 206.5, 406.0, 208.5, 407.0, 208.5, 411.5, 213.0, 411.5, 214.0, 417.0, 219.5, 418.0, 219.5, 422.0, 223.5, 423.0, 223.5, 426.0, 226.5, 427.0, 226.5, 429.0, 228.5, 430.0, 228.5, 431.0, 229.5, 432.0, 229.5, 433.0, 230.5, 434.0, 230.5, 438.5, 235.0, 438.5, 236.0, 440.5, 238.0, 440.5, 239.0, 441.5, 240.0, 441.5, 241.0, 442.5, 242.0, 442.5, 244.99999999999997, 443.5, 245.99999999999997, 443.5, 250.00000000000003, 444.5, 251.0, 444.5, 257.0, 445.5, 258.0, 445.5, 268.0, 446.5, 269.0, 446.5, 275.0, 445.5, 276.0, 445.5, 288.0, 444.5, 289.0, 444.5, 297.0, 443.5, 298.0, 443.5, 304.0, 442.5, 305.0, 442.5, 308.0, 441.5, 309.0, 441.5, 312.0, 440.5, 313.0, 440.5, 314.0, 438.5, 316.0, 438.5, 317.0, 435.0, 320.5, 434.0, 320.5, 428.5, 326.0, 428.5, 327.0, 425.5, 330.0, 425.5, 331.0, 417.0, 339.5, 416.0, 339.5, 412.5, 343.0, 412.5, 345.0, 411.5, 346.0, 411.5, 348.0, 410.5, 349.0, 410.5, 352.0, 409.5, 353.0, 409.5, 358.0, 408.5, 359.0, 408.5, 363.0, 407.5, 364.0, 407.5, 366.0, 406.5, 367.0, 406.5, 369.0, 405.5, 370.0, 405.5, 371.0, 404.5, 372.0, 404.5, 373.0, 403.5, 374.0, 403.5, 375.0, 395.5, 383.0, 395.5, 384.0, 394.0, 385.5, 393.0, 385.5, 392.0, 386.5, 391.0, 386.5, 390.0, 387.5, 389.0, 387.5, 388.0, 388.5, 386.0, 388.5, 385.0, 389.5, 383.0, 389.5, 382.0, 390.5, 379.0, 390.5, 378.0, 391.5, 375.0, 391.5, 374.0, 392.5, 372.0, 392.5, 371.0, 393.5, 369.0, 393.5, 368.0, 394.5, 367.0, 394.5, 366.0, 395.5, 365.0, 395.5, 364.0, 396.5, 363.0, 396.5, 362.0, 397.5, 361.0, 397.5, 360.0, 398.5, 358.0, 398.5, 357.0, 399.5, 355.0, 399.5, 354.0, 400.5, 351.0, 400.5, 350.0, 401.5, 349.0, 401.5, 348.0, 402.5, 346.0, 402.5, 344.0, 404.5, 343.0, 404.5, 342.0, 405.5, 341.0, 405.5, 339.0, 407.5, 337.0, 407.5, 336.0, 408.5, 332.0, 408.5, 331.0, 409.5, 326.0, 409.5, 325.0, 410.5, 319.0, 410.5, 318.0, 411.5, 306.0, 411.5, 305.0, 412.5, 295.0, 412.5, 294.0, 413.5, 286.0, 413.5, 285.0, 414.5, 279.0, 414.5, 278.0, 415.5, 273.0, 415.5, 272.0, 416.5, 264.0, 416.5, 263.0, 417.5, 241.0, 417.5, 240.0, 416.5, 202.0, 416.5, 201.0, 417.5, 196.0, 417.5, 195.0, 418.5, 183.0, 418.5]]}
#print (len(annotations['segmentation'][0]))