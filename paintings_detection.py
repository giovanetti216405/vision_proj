import numpy as np
import cv2 as cv
import paintings_retrieval as p_ret

# INPUT
# imgs = images of the database
# frame = current frame
def paintings_detection(imgs, frame):
    kernel = np.ones((3, 3), np.uint8)  # kernel for dilation
    paintings_coord = []  # list of all the detected paintings' coordinates in the current frame
    all_retrieved = []  # list of all the detected paintings' info (title, author, room, name, dif) in the current frame
    room = 0  # initialization of the room

    if frame is not None:
        # Image processing
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        canny = cv.Canny(blur, 33, 100)
        cv.imshow('canny', canny)

        # Dilation
        dil = cv.dilate(canny, kernel, iterations=1)
        dil = cv.dilate(dil, kernel, iterations=1)
        dil = cv.dilate(dil, kernel, iterations=1)
        cv.imshow('dilation', dil)

        # Find contours
        # Sort them for descending area
        # Consider only the first 10
        _, cnts, hier = cv.findContours(dil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours = sorted(cnts, key=cv.contourArea, reverse=True)[:10]

        for contour in contours:
            # Consider only the 4-sided polygon
            polygon = cv.approxPolyDP(contour, 0.1 * cv.arcLength(contour, True), True)
            if len(polygon) == 4:
                box = polygon

                # Consider only the polygons with area greater than 6000 (to avoid false positives)
                if cv.contourArea(box) > 7000:

                    cv.drawContours(frame, [box], 0, (0, 255, 0), 2)

                    box = box.reshape(4, 2)
                    sorted_box = np.zeros(box.shape, dtype="float32")

                    # ================================== PAINTING RECTIFICATION =====================================
                    # Sorted box (top-left, top-right, bottom-right, bottom-left)
                    sum = box.sum(axis=1)

                    sorted_box[0] = box[np.argmin(sum)]
                    sorted_box[2] = box[np.argmax(sum)]

                    diff = np.diff(box, axis=1)
                    sorted_box[1] = box[np.argmin(diff)]
                    sorted_box[3] = box[np.argmax(diff)]

                    smallest_x = 1000000
                    smallest_y = 1000000
                    largest_x = -1
                    largest_y = -1

                    for point in sorted_box:
                        if point[0] < smallest_x:
                            smallest_x = point[0]

                        if point[0] > largest_x:
                            largest_x = point[0]

                        if point[1] < smallest_y:
                            smallest_y = point[1]

                        if point[1] > largest_y:
                            largest_y = point[1]

                    # Compute width and height of the destination box
                    maxWidth = int(largest_x - smallest_x)
                    maxHeight = int(largest_y - smallest_y)

                    # Destination box
                    dst = np.array([
                        [0, 0],
                        [maxWidth, 0],
                        [maxWidth, maxHeight],
                        [0, maxHeight]], dtype="float32")

                    # Perspective Transorm matrix and rectification
                    transform = cv.getPerspectiveTransform(sorted_box, dst)
                    result = cv.warpPerspective(frame, transform, (0, 0))
                    rectified = result[0:maxHeight, 0:maxWidth]
                    cv.imshow('rectified', rectified)

                    # ================================== PAINTING RETRIVIAL =====================================
                    # retrieved = title, author, room, img-name, difference of the rectified painting
                    retrieved = p_ret.retrieval(rectified, imgs)

                    if retrieved == 0:
                        print("******No correspondence in the database.\n")
                    else:
                        all_retrieved.append(retrieved)
                        print(f'\t"{retrieved[0]}" is the painting of {retrieved[1]} in the room {retrieved[2]} ({retrieved[3]}).')
                        print(f'\t\t ' + "Coordinates: (" + str(smallest_x) + ", " + str(smallest_y) + "), (" + str(largest_x) + ", " + str(smallest_y) + "), (" + str(largest_x) + ", " + str(largest_y) + "), (" + str(smallest_x) + ", " + str(largest_y) + ")")

                        # Visualization of the painting's name above the bounding box
                        title = "{}".format(retrieved[0])
                        title = "'" + title[:25] + "'"
                        x = int(smallest_x)
                        y = int(smallest_y)
                        cv.putText(frame, title, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                        # Store the rectified painting's coordinates in the list
                        painting_data = (smallest_x, smallest_y, maxWidth, maxHeight)
                        paintings_coord.append(painting_data)

        # Choice of the room of the painting with the highest difference value
        if all_retrieved:
            sorted_for_diff = sorted(all_retrieved, key=lambda paint: paint[4], reverse=True)
            room = sorted_for_diff[0][2]

    return frame, paintings_coord, room