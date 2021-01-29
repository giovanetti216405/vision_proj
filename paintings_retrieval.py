import cv2
import csv

# Function for data.csv reading --> information extraction
# INPUT:
# n_img = index of the selected image of the database
# diff = difference between the amounts of good matches of the selected image and of the second in the list
def read_csv(n_img, diff):
    csv_data = []
    n_img = str(n_img)
    n_img = n_img.rjust(3, '0')
    n_img = n_img + '.png'
    with open('data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif row[3] == n_img:
                csv_data.append(row[0]) #Title
                csv_data.append(row[1]) #author
                csv_data.append(row[2]) #room
                csv_data.append(row[3]) #n_img
                csv_data.append(diff)  #difference
                return csv_data

# INPUT:
# rectified = rectified detected painting
# imgs = database images
def retrieval(rectified, imgs):
    img1 = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

    # SIFT:
    # keypoint_1 represents the array of keypoints for img1
    # descriptor_1 stores the array of descriptors for keypoint_1.
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)

    # initialization of the list containing descriptors matches
    matches = []

    # loop over all the images of the database for comparison
    for im in imgs:
        img2 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # SIFT:
        # keypoint_2 represents the array of keypoints for img2
        # descriptor_2 stores the array of descriptors for keypoint_2
        keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

        # MATCHING:
        # The descriptors are used to compare keypoints in the two images.
        # knnMatch returns the two nearest descriptors:
        # the 1st match is the closest neighbour and the 2nd match is the 2nd closest neighbour
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn_matches = bf.knnMatch(descriptors_1, descriptors_2, 2)

        # Filter matches using the Lowe's ratio test:
        # consider as good matches only those that have the distance ratio
        # between the 1st and the 2nd neighbours smaller than 0.7
        ratio_thresh = 0.7
        good_matches = []

        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        matches.append(good_matches)

    # num_good_matches = ordered list containing amounts of good matches for each DB image
    num_good_matches = []
    for i in range(len(matches)):
        num_good_matches.append(len(matches[i]))

    num_good_matches = sorted(num_good_matches, reverse=True)

    # If there isn't matches --> retrieved = 0 --> no similarities in DB found
    retrieved = 0
    if num_good_matches[0] != 0:
        # sorted_img_name = list containing the database images' indexes sorted acoording num_good_matches order
        sorted_img_name = sorted(range(len(matches)), key=lambda k: len(matches[k]), reverse=True)

        # THRESHOLD
        difference = num_good_matches[0] - num_good_matches[1]
        threshold = 6
        if difference > threshold:
            print("*****THE DETECTED PAINTING IS:")
        else:
            print("*****PROBABLY THE DETECTED PAINTING IS:")

        retrieved = read_csv(sorted_img_name[0], difference)

    return retrieved
